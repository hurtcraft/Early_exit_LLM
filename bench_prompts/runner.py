import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tqdm import tqdm

from bench_prompts.metrics import build_run_config, compare_to_baseline, summarize_run
from early_exit.generate import early_exit_generate
from early_exit.strategies.NoExitStrategy import NoExitStrategy
from early_exit.utils import expand_grid


class PromptGridRunner:
    """
    Lance automatiquement plusieurs combinaisons (stratégie x hyperparamètres)
    sur un même jeu de prompts custom.

    NoExitStrategy est TOUJOURS exécutée en premier et sert de baseline :
    chaque autre run est comparé à ses réponses, question par question.

    Fichiers écrits dans results_dir (un dossier par modèle) :
        runs/<Stratégie>_<params>.json   détail d'un run
        GRID_SUMMARY.json                résumé de tous les runs
    """

    def __init__(
        self,
        adapter,
        model_name: str,
        quant: str,
        questions: List[str],
        results_dir: Path,
        max_new_tokens: int = 30,
        use_chat_template: bool = True,
        text_key: str = "answer",               # texte généré (clé de _compute_metrics)
        exit_layers_key: str = "exit_layers",   # couche de sortie de chaque token
        tokens_key: Optional[str] = "tokens",   # ids des tokens générés
        verbose: bool = False,                  # affiche chaque token et le top-k de chaque couche
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.quant = quant
        self.questions = questions
        self.results_dir = Path(results_dir)
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        self.text_key = text_key
        self.exit_layers_key = exit_layers_key
        self.tokens_key = tokens_key
        self.verbose = verbose

        self._experiments: List[tuple] = []
        self.all_summaries: List[Dict[str, Any]] = []
        self.baseline_results: Optional[List[Any]] = None
        self.summary_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Construction des expériences
    # ------------------------------------------------------------------ #
    def add_strategy(
        self,
        strategy_cls: Type,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> "PromptGridRunner":
        """
        param_grid   : paramètres à faire varier  -> {"threshold": [0.1, 0.2]}
        fixed_params : paramètres constants       -> {"min_layer": 10}

        NoExitStrategy est ignorée ici : elle est lancée automatiquement
        en premier comme baseline.
        """
        if strategy_cls is NoExitStrategy:
            return self
        self._experiments.append((strategy_cls, param_grid or {}, fixed_params or {}))
        return self

    def _iter_runs(self):
        for strategy_cls, param_grid, fixed in self._experiments:
            for combo in expand_grid(param_grid):
                yield strategy_cls, {**fixed, **combo}

    def count_runs(self) -> int:
        """Nombre total de runs, baseline comprise."""
        return 1 + sum(len(expand_grid(g)) for _, g, _ in self._experiments)

    # ------------------------------------------------------------------ #
    # Exécution
    # ------------------------------------------------------------------ #
    def _build_output(self, strategy, results: List[Any], compare: bool = True) -> Dict[str, Any]:
        config = build_run_config(
            self.adapter, strategy, self.model_name, self.quant,
            self.use_chat_template, self.max_new_tokens,
        )

        output = {
            "config": config,
            "summary": summarize_run(results, self.adapter.num_layers),
            "results": results,
        }

        if compare and self.baseline_results is not None:
            output["comparison"] = compare_to_baseline(
                self.baseline_results, results, self.adapter.num_layers,
                self.text_key, self.exit_layers_key, self.tokens_key,
            )
        return output

    def _save(self, output: Dict[str, Any], strategy, params: Dict[str, Any]) -> Path:
        runs_dir = self.results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # le dossier indique déjà la date et le modèle : nom = stratégie + paramètres
        params_str = "_".join(
            f"{k}{str(v).replace('.', 'p')}"
            for k, v in params.items()
            if k != "tokenizer"
        )
        name = strategy.__class__.__name__ + (f"_{params_str}" if params_str else "")
        path = runs_dir / f"{name}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        return path

    def _build_runs(self, stop_on_error: bool) -> List[Dict[str, Any]]:
        """Instancie les stratégies ; la baseline NoExitStrategy est toujours en tête."""
        runs = [{"name": "NoExitStrategy", "params": {}, "label": "NoExitStrategy()",
                 "strategy": NoExitStrategy(), "results": [], "error": None}]

        for strategy_cls, params in self._iter_runs():
            run = {"name": strategy_cls.__name__, "params": params,
                   "label": f"{strategy_cls.__name__}({self._fmt_params(params)})",
                   "strategy": None, "results": [], "error": None}
            try:
                run["strategy"] = strategy_cls(**params)
            except Exception as e:
                self._report_error(run, e)
                if stop_on_error:
                    raise
            runs.append(run)
        return runs

    @staticmethod
    def _report_error(run: Dict[str, Any], e: Exception) -> None:
        tqdm.write(f"!! ERREUR {run['label']} : {e}")
        traceback.print_exc()
        run["error"] = str(e)

    def run_all(self, stop_on_error: bool = False) -> List[Dict[str, Any]]:
        runs = self._build_runs(stop_on_error)
        baseline = runs[0]
        n_q = len(self.questions)
        print(f"=== {len(runs)} runs x {n_q} questions (baseline incluse) ===\n")

        # Boucle question -> runs : les forwards d'un même préfixe sont partagés
        # entre tous les runs via trace_cache, vidé à chaque question.
        with tqdm(total=len(runs) * n_q, desc="Génération", unit="gen") as bar:
            for qi, question in enumerate(self.questions):
                trace_cache: Dict[tuple, Any] = {}
                for run in runs:
                    bar.set_postfix_str(f"q{qi + 1}/{n_q} {run['label']}")
                    if run["error"] is None:
                        try:
                            run["results"].append(early_exit_generate(
                                self.adapter,
                                run["strategy"],
                                question,
                                max_new_tokens=self.max_new_tokens,
                                use_chat_template=self.use_chat_template,
                                verbose=self.verbose,
                                trace_cache=trace_cache,
                            ))
                        except Exception as e:
                            # échec de la baseline = arrêt (rien à comparer sans elle)
                            if run is baseline or stop_on_error:
                                raise
                            self._report_error(run, e)
                    bar.update(1)

        self.baseline_results = baseline["results"]
        for run in runs:
            clean_params = {k: v for k, v in run["params"].items() if k != "tokenizer"}
            if run["error"] is not None:
                self.all_summaries.append({
                    "strategy": run["name"],
                    "params": clean_params,
                    "error": run["error"],
                })
                continue

            is_baseline = run is baseline
            output = self._build_output(run["strategy"], run["results"], compare=not is_baseline)
            path = self._save(output, run["strategy"], run["params"])

            entry = {
                "strategy": run["name"],
                "params": clean_params,
                "summary": output["summary"],
                "file": str(path),
            }
            if not is_baseline:
                # métriques agrégées vs baseline (le détail par question est dans le JSON du run)
                entry["comparison"] = output["comparison"]["summary"]
            self.all_summaries.append(entry)
            print(f"{run['label']} -> sauvegardé : {path.name}")

        self._save_global_summary()
        return self.all_summaries

    @staticmethod
    def _fmt_params(params: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items() if k != "tokenizer")

    def _save_global_summary(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / "GRID_SUMMARY.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.all_summaries, f, indent=2, ensure_ascii=False, default=str)
        self.summary_path = path   # lu par run_prompts.py pour générer les plots
        print(f"Résumé global : {path}")
