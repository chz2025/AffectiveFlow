#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def softmax(score_map: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not score_map:
        return {}
    if temperature <= 0:
        temperature = 1.0
    max_score = max(score_map.values())
    exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in score_map.items()}
    total = sum(exp_scores.values())
    if total == 0:
        return {k: 1.0 / len(exp_scores) for k in exp_scores}
    return {k: v / total for k, v in exp_scores.items()}


def format_memory(memory: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for item in memory:
        for key, value in item.items():
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def format_chat_history(history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in history:
        role = turn.get("role", "unknown").capitalize()
        strategy = turn.get("strategy")
        prefix = f"{role}" if not strategy else f"{role} [{strategy}]"
        content = turn.get("content", "")
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def parse_strategy_output(raw: str, strategies: List[str]) -> Dict[str, float]:
    # 尝试解析 JSON（有无反引号都行）
    match = re.search(r"`({.*})`", raw, re.DOTALL) or re.search(r"({.*})", raw, re.DOTALL)
    parsed: Dict[str, Any] = {}
    if match:
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            parsed = {}
    # 失败则用正则解析 key:value
    if not parsed:
        parsed = {}
        for m in re.finditer(r"\"?([^\":]+?)\"?\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw):
            parsed[m.group(1).strip()] = m.group(2)
    # 对齐策略名（去掉括号缩写）
    scores: Dict[str, float] = {}
    for name in strategies:
        for key, val in parsed.items():
            base = re.sub(r"\s*\(.*\)$", "", key).strip()  # 去掉 "(EV)" 之类的缩写
            if base == name:
                try:
                    scores[name] = float(val)
                except (TypeError, ValueError):
                    pass
                break
    return scores


def parse_reward_output(raw: str, metric_names: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    # 形式1: 数字 “1: 5”
    for m in re.finditer(r"(\d+)\s*:\s*([0-5](?:\.\d+)?)", raw):
        idx = int(m.group(1)) - 1
        val = float(m.group(2))
        if 0 <= idx < len(metric_names):
            scores[metric_names[idx]] = val
    # 形式2: 名称 “<Empathy>: <5>” 或 “Empathy: 5”
    for m in re.finditer(r"[<\[]?\s*([A-Za-z_]+)\s*[>\]]?\s*:\s*[<\[]?([0-5](?:\.\d+)?)", raw):
        name = m.group(1).replace("_", " ").strip()
        val = float(m.group(2))
        for metric in metric_names:
            if metric.lower().replace(" ", "") == name.lower().replace(" ", ""):
                scores[metric] = val
                break
    return scores


def compute_weighted_reward(
    reward_scores: Dict[str, float], weights: Dict[str, float]
) -> float:
    weighted_sum = 0.0
    for name, weight in weights.items():
        if name not in reward_scores:
            continue
        weighted_sum += reward_scores[name] * weight
    return weighted_sum


class Logger:
    _main_logger: Optional[logging.Logger] = None
    _prompt_logger: Optional[logging.Logger] = None

    @classmethod
    def init(cls, log_path: pathlib.Path, enable_console: bool = False) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        main_logger = logging.getLogger("ex_tree")
        main_logger.setLevel(logging.INFO)
        main_logger.handlers.clear()
        main_logger.addHandler(file_handler)
        if enable_console:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter("%(message)s"))
            main_logger.addHandler(console)
        main_logger.propagate = False
        cls._main_logger = main_logger

        prompt_logger = logging.getLogger("ex_tree.prompt")
        prompt_logger.setLevel(logging.INFO)
        prompt_logger.handlers.clear()
        prompt_logger.addHandler(file_handler)
        prompt_logger.propagate = False
        cls._prompt_logger = prompt_logger

    @classmethod
    def get(cls) -> logging.Logger:
        if cls._main_logger is None:
            raise RuntimeError("Logger not initialized. Call Logger.init first.")
        return cls._main_logger

    @classmethod
    def prompt_logger(cls) -> logging.Logger:
        if cls._prompt_logger is None:
            raise RuntimeError("Logger not initialized. Call Logger.init first.")
        return cls._prompt_logger

    @classmethod
    def log_prompt(cls, role: str, prompt: str, output: str, offline: bool) -> None:
        mode = "offline" if offline else "online"
        cls.prompt_logger().info(
            "role=%s mode=%s\nPROMPT:\n%s\nOUTPUT:\n%s\n---",
            role,
            mode,
            prompt,
            output,
        )


@dataclass
class TreeNode:
    node_id: str
    depth: int
    history: List[Dict[str, Any]]
    strategy: Optional[str] = None
    supporter_response: Optional[str] = None
    seeker_response: Optional[str] = None
    # strategy_scores: Dict[str, float] = field(default_factory=dict)
    strategy_probs: Dict[str, float] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    # reward_trace: List[float] = field(default_factory=list)
    children: Dict[str, "TreeNode"] = field(default_factory=dict)
    is_terminal: str = ""
    action_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "strategy": self.strategy,
            "supporter_response": self.supporter_response,
            "seeker_response": self.seeker_response,
            # "strategy_scores": self.strategy_scores,
            "strategy_probs": self.strategy_probs,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward(),
            # "reward_trace": self.reward_trace,
            "history": self.history,
            "is_terminal": self.is_terminal,
            "action_stats": self.action_stats,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


class Stats:
    prompt_total: Counter = Counter()
    prompt_success: Counter = Counter()
    tree_node_counts: List[int] = []
    scenes_processed: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.prompt_total.clear()
        cls.prompt_success.clear()
        cls.tree_node_counts.clear()
        cls.scenes_processed = 0

    @classmethod
    def record_prompt(cls, role: str, success: bool) -> None:
        cls.prompt_total[role] += 1
        if success:
            cls.prompt_success[role] += 1

    @classmethod
    def record_tree(cls, node_count: int) -> None:
        cls.tree_node_counts.append(node_count)
        cls.scenes_processed += 1

    @classmethod
    def total_nodes(cls) -> int:
        return sum(cls.tree_node_counts)

    @classmethod
    def prompt_postfix(cls) -> Dict[str, str]:
        out = {}
        for role in ["strategy", "supporter", "seeker", "reward"]:
            total = cls.prompt_total.get(role, 0)
            succ = cls.prompt_success.get(role, 0)
            out[role] = f"{succ}/{total}"
        return out

    @classmethod
    def summary_line(
        cls,
        processed: int,
        total: int,
        last_nodes: int,
        epoch: Optional[int] = None,
        epoch_total: Optional[int] = None,
        batch: Optional[int] = None,
        batch_total: Optional[int] = None,
    ) -> str:
        header = []
        if epoch is not None and epoch_total is not None:
            header.append(f"Epoch {epoch}/{epoch_total}")
        if batch is not None and batch_total is not None:
            header.append(f"Batch {batch}/{batch_total}")
        header_text = " ".join(header) if header else "Progress"

        parts = [
            header_text,
            f"Processed {processed}/{total} scenes",
            f"trees built: {processed}",
            f"last tree nodes: {last_nodes}",
            f"total nodes: {cls.total_nodes()}",
        ]
        role_parts = []
        for role in ["strategy", "supporter", "seeker", "reward"]:
            total_calls = cls.prompt_total.get(role, 0)
            succ_calls = cls.prompt_success.get(role, 0)
            role_parts.append(f"{role}: {succ_calls}/{total_calls}")
        parts.append("prompt calls (success/total): " + ", ".join(role_parts))
        return " | ".join(parts)


class BatchProgress:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.count = 0
        try:
            from tqdm import tqdm  # type: ignore

            self.bar = tqdm(
                total=total,
                desc=desc,
                leave=True,
                dynamic_ncols=True,
                disable=False,
            )
        except Exception:
            self.bar = None
            print(desc)

    def update(self, postfix: Dict[str, str]) -> None:
        self.count += 1
        if self.bar:
            self.bar.update(1)
        else:
            line1 = f"[{self.count}/{self.total}]"
            line2 = " ".join(f"{k}={v}" for k, v in postfix.items())
            print(line1)
            print(line2)

    def finalize(self, summary_line: Optional[str] = None) -> None:
        if self.bar:
            self.bar.close()
            print()
        if summary_line:
            print(summary_line)


class LLMInterface:
    def __init__(
        self,
        cfg: Dict[str, Any],
        role: str,
        offline: bool = False,
        known_strategies: Optional[List[str]] = None,
    ):
        self.role = role
        self.offline = offline
        self.model_name = cfg.get("model_name")
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.9)
        self.max_tokens = cfg.get("max_new_tokens", 256)
        self.client = None
        self.known_strategies = known_strategies or []

        if not self.offline and OpenAI is not None:
            api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
            api_base = cfg.get("api_base")
            if api_key and api_base:
                try:
                    self.client = OpenAI(base_url=api_base, api_key=api_key)
                except Exception:
                    self.client = None
            else:
                self.offline = True
        else:
            self.offline = True

    def generate(self, prompt: str) -> str:
        if self.offline or self.client is None:
            output = self._offline_generate(prompt)
            Logger.log_prompt(self.role, prompt, output, offline=True)
            Stats.record_prompt(self.role, success=True)
            return output
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            output = completion.choices[0].message.content.strip()
            Logger.log_prompt(self.role, prompt, output, offline=False)
            Stats.record_prompt(self.role, success=True)
            return output
        except Exception as exc:
            Logger.get().warning("LLM call failed for role=%s, falling back offline. err=%s", self.role, exc)
            output = self._offline_generate(prompt)
            Logger.log_prompt(self.role, prompt, output, offline=True)
            Stats.record_prompt(self.role, success=False)
            return output

    def _offline_generate(self, prompt: str) -> str:
        random.seed(hash(prompt) % (2**32))
        if self.role == "strategy":
            strategies = self.known_strategies or re.findall(r"strategy\": \"([^\"]+)\"", prompt)
            if not strategies:
                strategies = [f"Strategy-{i}" for i in range(8)]
            scores = {name: random.randint(4, 9) for name in strategies}
            reasoning = "Offline mode: heuristic scoring based on prompt keywords."
            return f"{reasoning}\n`{json.dumps(scores, ensure_ascii=False)}`"
        if self.role == "supporter":
            last_line = ""
            if "History:" in prompt:
                history_lines = [ln for ln in prompt.split("History:")[-1].strip().splitlines() if ln.strip()]
                last_line = history_lines[-1] if history_lines else ""
            snippet = last_line.split(":")[-1].strip()
            msg = snippet[:12] if snippet else "I get it"
            return f"I hear you, {msg}. I'm here with you."
        if self.role == "seeker":
            return "Thanks, still figuring this out."
        if self.role == "reward":
            lines = [f"{idx}: {random.randint(3,5)}" for idx in range(1, 5)]
            return "\n".join(lines)
        return "Placeholder response."


class MCTSBuilder:
    def __init__(
        self,
        cfg: Dict[str, Any],
        prompts: Dict[str, str],
        strategies: List[Dict[str, str]],
        evaluation_metrics: Dict[str, Any],
        offline: bool = False,
    ):
        self.cfg = cfg
        self.prompts = prompts
        self.strategies = strategies
        self.strategy_names = [s["strategy"] for s in strategies]
        self.strategy_detail_map = {s["strategy"]: s["strategy_detail"] for s in strategies}
        self.evaluation_metrics = evaluation_metrics.get("evaluation_metrics", [])
        self.metric_names = [item["dimension"] for item in self.evaluation_metrics]
        self.evaluation_criteria_text = json.dumps(
            self.evaluation_metrics, ensure_ascii=False, indent=2
        )
        self.strategy_block = self._build_strategy_block()
        self.weights = cfg.get("mcts", {}).get("reward_weights", {})
        mcts_cfg = cfg.get("mcts", {})
        self.exploration_constant = float(mcts_cfg.get("exploration_constant", 1.0))
        self.rollout_steps = int(mcts_cfg.get("rollout_steps", 2))
        self.max_depth = int(mcts_cfg.get("max_depth", 10))
        self.simulations = int(mcts_cfg.get("simulations_per_tree", 150))
        self.strategy_temperature = float(mcts_cfg.get("strategy_temperature", 1.0))
        self.ucb_visit_offset = float(mcts_cfg.get("ucb_visit_offset", 1.0))
        self.rho_kappa = float(mcts_cfg.get("rho_kappa", 1.0))
        self.selection_lambda = float(mcts_cfg.get("selection_lambda", 1.0))
        self.reward_norm = float(mcts_cfg.get("reward_normalization", 1.0))

        models_cfg = cfg.get("models", {})
        self.strategy_llm = LLMInterface(
            models_cfg.get("strategy", {}), "strategy", offline, known_strategies=self.strategy_names
        )
        self.supporter_llm = LLMInterface(models_cfg.get("supporter", {}), "supporter", offline)
        self.seeker_llm = LLMInterface(models_cfg.get("seeker", {}), "seeker", offline)
        reward_cfg = models_cfg.get("rewarde") or models_cfg.get("reward", {})
        self.reward_llm = LLMInterface(reward_cfg, "reward", offline)
    
    def _evaluate_leaf_node(self, node: TreeNode) -> float:
        reward_scores = self.score_reward(node.history)
        value = compute_weighted_reward(reward_scores, self.weights) / max(1.0, self.reward_norm)
        return value
    
    def _build_strategy_block(self) -> str:
        lines: List[str] = []
        for item in self.strategies:
            lines.append(
                f"- {item['strategy']} ({item['abbreviation']}): {item['strategy_detail']}"
            )
        return "\n".join(lines)

    def build_tree_for_scene(self, sample: Dict[str, Any]) -> TreeNode:
        memory_text = sample.get("memory", [])
        if not memory_text:
            raise ValueError("Sample missing memory field.")
        
        # 兼容性处理：尝试提取第一条消息
        try:
            first_msg_val = next(iter(memory_text[0].values()))
        except (IndexError, AttributeError):
            first_msg_val = "Hello."

        root_history: List[Dict[str, Any]] = [{"role": "seeker", "content": first_msg_val}]
        root = TreeNode(node_id="root", depth=0, history=root_history)
        
        # 初始化根节点的策略概率
        probs = self.score_strategies(root)
        root.strategy_probs = probs

        for _ in range(self.simulations):
            node = root
            # Path 记录 (父节点, 采取的动作)。不记录终止节点的 None 动作。
            path: List[Tuple[TreeNode, str]] = []

            while True:
                # 1. 终止或截断判断
                # 如果当前节点已经是终止节点，或者深度达到限制，直接评估
                if node.is_terminal == "<ok>" or node.depth >= self.max_depth:
                    if node.visits > 0:
                        value = node.mean_reward()
                    else:
                        value = self._evaluate_leaf_node(node)
                    self.backpropagate(path, value)
                    break
                if node.is_terminal == "</end/>":
                    value = node.mean_reward()
                    self.backpropagate(path, value)
                    break
                # 2. 选择动作 (Selection)
                strategy = self.select_action(node)

                # 3. 扩展 (Expansion)
                if strategy not in node.children:
                    child = self.expand(node, sample, strategy)
                    # 记录这一步边到路径
                    path.append((node, strategy))
                    # 4. 模拟 (Simulation)
                    value = self.simulate(child, sample)
                    # 5. 回传
                    self.backpropagate(path, value)
                    # 本次 MCTS 迭代结束，跳出 while 循环，进行下一次 simulation
                    break

                # 5. 下探 (Traverse)
                # 子节点已存在，记录路径并继续向下搜索
                path.append((node, strategy))
                node = node.children[strategy]
                # 继续 while 循环...

        return root
    

    def select_action(self, node: TreeNode) -> str:
        # Formula 6/12: N(s) = Σ_a N(s,a).
        parent_visits = float(node.visits)
        # Formula 5: ρ = κ / (κ + log(N(s)+1)).
        rho = self.rho_kappa / (1+ math.log(parent_visits + 1))

        scores: Dict[str, float] = {}
        for strategy in self.strategy_names:
            prior = node.strategy_probs.get(strategy, 0.0)
            stats = node.action_stats.get(strategy, {"visits": 0.0, "total_reward": 0.0, "q": 0.0})
            n_sa = stats.get("visits", 0.0)
            q_sa = stats.get("q", 0.0)
            denom = self.ucb_visit_offset + n_sa
            # Formula 7: softmax over Q + λ * prior * sqrt(N(s)) / (ucb_visit_offset + N(s,a)).
            explore = self.selection_lambda * prior * math.sqrt(parent_visits / denom)
            scores[strategy] = q_sa + explore
        soft_probs = softmax(scores)
        
        mix_probs = {
            s: (1 - rho) * soft_probs.get(s, 0.0) + (0.1 * rho)
            for s in self.strategy_names
        }
        # Greedy selection: pick the strategy with highest mixed probability.
        return max(mix_probs, key=mix_probs.get)

    def expand(self, node: TreeNode, sample: Dict[str, Any], strategy: str) -> TreeNode:
        explored = set(node.children.keys())
        if strategy in explored:
            return node.children[strategy]
        supporter_resp = self.generate_supporter(strategy, node.history)
        new_history = list(node.history)
        new_history.append({"role": "supporter", "content": supporter_resp, "strategy": strategy})
        seeker_resp = self.generate_seeker(sample, new_history)
        new_history.append({"role": "seeker", "content": seeker_resp})
        child_id = f"{node.node_id}/{len(node.children)}"
        s_trim = seeker_resp.strip()
        terminal = ""
        if s_trim.endswith("<ok>"):
            terminal = "<ok>"
        if s_trim.strip()=="</end/>":
            terminal = "</end/>"
        child = TreeNode(
            node_id=child_id,
            depth=node.depth + 1,
            history=new_history,
            strategy=strategy,
            supporter_response=supporter_resp,
            seeker_response=seeker_resp,
            # is_terminal=seeker_resp.strip() == "</end/>",
            is_terminal=terminal,

        )
        probs = self.score_strategies(child)
        child.strategy_probs = probs
        node.children[strategy] = child
        return child

    def simulate(self, node: TreeNode, sample: Dict[str, Any]) -> float:
       
        rollout_history = list(node.history)
        total_reward = 0.0
        steps_run = 0
        terminal_value = 1  # default

        for step in range(self.rollout_steps):
            steps_run += 1
            if step == 0 and node.strategy_probs:
                probs = node.strategy_probs
            else:
                probs = self.score_strategies_with_history(rollout_history)
            if not probs:
                break
            strategy = max(probs.items(), key=lambda kv: kv[1])[0]
            
            # 1. 生成 Supporter 回复
            supporter_resp = self.generate_supporter(strategy, rollout_history)
            rollout_history.append({"role": "supporter", "content": supporter_resp, "strategy": strategy})
            
            # 2. 计算当前步的 Reward
            reward_scores = self.score_reward(rollout_history)
            reward_value = compute_weighted_reward(reward_scores, self.weights) / max(1.0, self.reward_norm)
            total_reward += reward_value
            
            # 3. 生成 Seeker 回复
            seeker_resp = self.generate_seeker(sample, rollout_history)
            rollout_history.append({"role": "seeker", "content": seeker_resp})
            
            # 4. 检查是否结束
            s_trim = seeker_resp.strip()
            current_depth = node.depth + step + 1
            if s_trim.endswith("<ok>") or current_depth >= self.max_depth:
                terminal_value = 1.0
                break
            if s_trim == "</end/>":
                terminal_value = 0.0
                break
        # 返回平均步长奖励
        return terminal_value * total_reward / max(1, steps_run)

    def backpropagate(self, path: List[Tuple[TreeNode, Optional[str]]], reward: float) -> None:
        for node, action in path:
            # 更新节点统计
            node.visits += 1
            node.total_reward += reward
            
            # 更新边（动作）统计
            if action:
                stats = node.action_stats.setdefault(action, {"visits": 0.0, "total_reward": 0.0, "mean_reward": 0.0, "q": 0.0})
                stats["visits"] += 1
                stats["total_reward"] += reward
                stats["mean_reward"] = stats["total_reward"] / stats["visits"]
                # 增量更新 Q 值
                # stats["q"] += (stats["total_reward"]  - stats["q"]) / stats["visits"]
                stats["q"] += (reward - stats["q"]) / stats["visits"]

    def score_strategies(self, node: TreeNode) -> Dict[str, float]:
        if node.strategy_probs:
            return node.strategy_probs
        prompt = self.prompts["strategy prompt"].format(
            all_stratrgy_descriptions=self.strategy_block,
            chat_history=format_chat_history(node.history),
        )
        raw = self.strategy_llm.generate(prompt)
        scores = parse_strategy_output(raw, self.strategy_names)
        for name in self.strategy_names:
            scores.setdefault(name, 0.0)
        probs = softmax(scores, temperature=self.strategy_temperature)
        # probs = scores
        return probs
    
    def score_strategies_with_history(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        dummy_node = TreeNode(node_id="rollout", depth=0, history=history)
        return self.score_strategies(dummy_node)

    def generate_supporter(self, strategy: str, history: List[Dict[str, Any]]) -> str:
        detail = self.strategy_detail_map.get(strategy, "")
        prompt = self.prompts["supporter prompt"].format(
            strategy=strategy,
            strategy_detail=detail,
            chat_history=format_chat_history(history),
        )
        return self.supporter_llm.generate(prompt).strip()

    def generate_seeker(self, sample: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        prompt = self.prompts["seeker prompt"].format(
            scene=sample.get("scene", ""),
            description=sample.get("description", ""),
            memory=format_memory(sample.get("memory", [])),
            chat_history=format_chat_history(history),
        )
        return self.seeker_llm.generate(prompt).strip()

    def score_reward(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        prompt = self.prompts["reward prompt"].format(
            evaluation_criteria=self.evaluation_criteria_text,
            chat_history=format_chat_history(history),
        )
        raw = self.reward_llm.generate(prompt)
        return parse_reward_output(raw, self.metric_names)


def load_prompts(path: pathlib.Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        prompt_items = json.load(f)
    prompts: Dict[str, str] = {}
    for item in prompt_items:
        prompts.update(item)
    return prompts


def load_dataset(path: pathlib.Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(samples) >= limit:
                break
    return samples


def count_nodes(node: TreeNode) -> int:
    total = 1
    for child in node.children.values():
        total += count_nodes(child)
    return total


def chunked(items: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        batch_size = 1
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def print_run_table(rows: List[Tuple[str, str]]) -> None:
    if not rows:
        return
    key_width = max(len(k) for k, _ in rows)
    val_width = max(len(str(v)) for _, v in rows)
    sep = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
    print(sep)
    for k, v in rows:
        print(f"| {k.ljust(key_width)} | {str(v).ljust(val_width)} |")
    print(sep)


def main() -> None:
    config_path = pathlib.Path("configs/train_emoflow.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_cfg = cfg.get("data", {})
    processed_dir = pathlib.Path(data_cfg.get("processed_dir", "data/processed/extes"))
    split = data_cfg.get("split", "train")
    dataset_path = processed_dir / f"{split}.jsonl"

    base_output = pathlib.Path(cfg.get("output", {}).get("tree_path", "data/processed/extes/Ex_Tree.jsonl"))
    base_log = pathlib.Path(cfg.get("output", {}).get("log_path", "logs/ex_tree.log"))

    # Inject split into filenames for all generated artifacts
    base_output = base_output.with_name(f"{base_output.stem}_{split}{base_output.suffix}")
    base_log = base_log.with_name(f"{base_log.stem}_{split}{base_log.suffix}")

    output_path = base_output.with_name(f"{base_output.stem}_{ts}{base_output.suffix}")
    log_path = base_log.with_name(f"{base_log.stem}_{ts}{base_log.suffix}")
    meta_path = pathlib.Path("analyze/tree_paths.json")
    scene_limit_cfg = cfg.get("output", {}).get("scene_limit")
    scene_limit = int(scene_limit_cfg) if scene_limit_cfg is not None else None
    offline = bool(cfg.get("run", {}).get("offline", False))

    prompts = load_prompts(pathlib.Path("data/prompt.json"))
    strategies = json.loads(pathlib.Path("data/strategies.json").read_text())
    evaluation_metrics = json.loads(pathlib.Path("data/evaluation_metrics.json").read_text())
    samples = load_dataset(dataset_path, limit=scene_limit)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 1))

    Logger.init(log_path, enable_console=False)
    Stats.reset()
    logger = Logger.get()
    print_run_table(
        [
            ("config", str(config_path)),
            ("output", str(output_path)),
            ("log", str(log_path)),
            ("offline", offline),
            ("scene_limit", scene_limit if scene_limit is not None else "all"),
            ("epochs", epochs),
            ("batch_size", batch_size),
        ]
    )

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"tree_path_train": str(output_path), "log_path": str(log_path)}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    builder = MCTSBuilder(
        cfg=cfg,
        prompts=prompts,
        strategies=strategies,
        evaluation_metrics=evaluation_metrics,
        offline=offline,
    )

    global_idx = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for epoch in range(epochs):
            batches = chunked(samples, batch_size)
            for batch_idx, batch in enumerate(batches):
                desc = f"epoch {epoch+1}/{epochs} batch {batch_idx+1}/{len(batches)} (size={len(batch)})"
                batch_bar = BatchProgress(len(batch), desc=desc)
                last_nodes = 0
                for sample in batch:
                    logger.debug(
                        "Building tree for scene #%s (epoch %s): %s",
                        global_idx + 1,
                        epoch + 1,
                        sample.get("scene"),
                    )
                    tree_root = builder.build_tree_for_scene(sample)
                    node_count = count_nodes(tree_root)
                    Stats.record_tree(node_count)
                    tree_obj = {
                        "epoch": epoch + 1,
                        "scene_index": global_idx,
                        "scene": sample.get("scene"),
                        "description": sample.get("description"),
                        "root": tree_root.to_dict(),
                    }
                    # out_f.write(json.dumps(tree_obj, ensure_ascii=False) + "\n")
                    out_f.write(json.dumps(tree_obj, ensure_ascii=False, indent=2) + "\n")
                    global_idx += 1
                    last_nodes = node_count
                    batch_bar.update(Stats.prompt_postfix())
                summary = Stats.prompt_postfix()
                summary_line = " ".join(
                    ["last_nodes=" + str(last_nodes)]
                    + [f"{k}={v}" for k, v in summary.items()]
                )
                batch_bar.finalize(summary_line)
    logger.info("Saved Ex_Tree to %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)