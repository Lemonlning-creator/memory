import json
import os

class TrustManager:
    def __init__(self, file_path="data/trust/trust_data.jsonl"):
        self.file_path = file_path
        self.current_trust = self._load_last_trust()

    def _load_last_trust(self):
        """从本地 jsonl 读取最新的信任值"""
        if not os.path.exists(self.file_path):
            return 0  # 初始值设定为 0 (怀疑阶段)
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    return 0
                last_line = json.loads(lines[-1])
                return last_line.get("trust_score", 0)
        except Exception:
            return 0

    def update_trust(self, user_input, behavior_score):
        """
        更新信任值并保存。behavior_score 由外部逻辑判断得出（正值为信任，负值为冒犯）
        """
        self.current_trust = max(0, min(100, self.current_trust + behavior_score))
        
        # 记录到本地 jsonl
        record = {
            "user_input": user_input,
            "change": behavior_score,
            "trust_score": self.current_trust
        }
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return self.current_trust

    def get_relationship_stage(self):
        """根据分值返回关系阶段名称"""
        if self.current_trust < 30:
            return "Initial (Suspicion)"
        elif self.current_trust < 80:
            return "Process (Utilization)"
        else:
            return "Final (Symbiosis)"