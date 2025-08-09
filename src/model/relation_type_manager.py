"""
关系类型管理器 - 管理关系类型的映射和编码
"""

from typing import Dict, List, Optional
import json
import os


class RelationTypeManager:
    """
    管理关系类型的双向映射：字符串 <-> 整数ID
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.rel_to_id: Dict[str, int] = {}
        self.id_to_rel: Dict[int, str] = {}
        self.next_id = 0
        self.vocab_path = vocab_path
        
        # 预定义一些常见关系类型
        self._init_default_relations()
        
        # 如果提供了词汇表路径，尝试加载
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
    
    def _init_default_relations(self):
        """初始化默认关系类型"""
        default_relations = [
            "NO_RELATION",  # 0: 无关系
            "PERSON_OF",    # 1: 人物关系
            "LOCATED_IN",   # 2: 位置关系
            "PART_OF",      # 3: 部分关系
            "MEMBER_OF",    # 4: 成员关系
            "BORN_IN",      # 5: 出生地
            "DIED_IN",      # 6: 死亡地
            "FOUNDED_BY",   # 7: 创立者
            "WORKS_FOR",    # 8: 工作关系
            "MARRIED_TO",   # 9: 婚姻关系
            "PARENT_OF",    # 10: 父母关系
            "CHILD_OF",     # 11: 子女关系
            "SIBLING_OF",   # 12: 兄弟姐妹关系
            "EDUCATED_AT",  # 13: 教育关系
            "NATIONALITY",  # 14: 国籍
            "OCCUPATION",   # 15: 职业
            "CAPITAL_OF",   # 16: 首都关系
            "CURRENCY_OF",  # 17: 货币关系
            "LANGUAGE_OF",  # 18: 语言关系
            "RELIGION_OF",  # 19: 宗教关系
        ]
        
        for rel in default_relations:
            self.add_relation(rel)
    
    def add_relation(self, relation: str) -> int:
        """
        添加新的关系类型，返回其ID
        Args:
            relation: 关系类型字符串
        Returns:
            int: 关系类型的ID
        """
        if relation not in self.rel_to_id:
            rel_id = self.next_id
            self.rel_to_id[relation] = rel_id
            self.id_to_rel[rel_id] = relation
            self.next_id += 1
            return rel_id
        return self.rel_to_id[relation]
    
    def get_id(self, relation: str) -> int:
        """
        获取关系类型的ID，如果不存在则添加
        Args:
            relation: 关系类型字符串
        Returns:
            int: 关系类型的ID
        """
        return self.add_relation(relation)
    
    def get_relation(self, rel_id: int) -> Optional[str]:
        """
        根据ID获取关系类型字符串
        Args:
            rel_id: 关系类型ID
        Returns:
            str or None: 关系类型字符串
        """
        return self.id_to_rel.get(rel_id)
    
    def get_all_relations(self) -> List[str]:
        """获取所有关系类型"""
        return list(self.rel_to_id.keys())
    
    def get_vocab_size(self) -> int:
        """获取关系类型词汇表大小"""
        return len(self.rel_to_id)
    
    def save_vocab(self, path: str):
        """保存关系类型词汇表"""
        vocab_data = {
            'rel_to_id': self.rel_to_id,
            'id_to_rel': {str(k): v for k, v in self.id_to_rel.items()},  # JSON需要字符串键
            'next_id': self.next_id
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved relation vocabulary to {path}")
    
    def load_vocab(self, path: str):
        """加载关系类型词汇表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.rel_to_id = vocab_data['rel_to_id']
            self.id_to_rel = {int(k): v for k, v in vocab_data['id_to_rel'].items()}  # 转换键为整数
            self.next_id = vocab_data['next_id']
            
            print(f"✅ Loaded relation vocabulary from {path} ({len(self.rel_to_id)} relations)")
            
        except Exception as e:
            print(f"⚠️ Failed to load relation vocabulary from {path}: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """获取关系类型统计信息"""
        return {
            'total_relations': len(self.rel_to_id),
            'next_available_id': self.next_id
        }
    
    def merge_from_spacy_labels(self, spacy_labels: List[str]):
        """
        从SpaCy标签中合并关系类型
        Args:
            spacy_labels: SpaCy识别的标签列表
        """
        # 将SpaCy的实体标签转换为关系类型
        spacy_to_relation_map = {
            'PERSON': 'PERSON_RELATION',
            'ORG': 'ORGANIZATION_RELATION', 
            'GPE': 'LOCATION_RELATION',
            'LOC': 'LOCATION_RELATION',
            'DATE': 'TEMPORAL_RELATION',
            'TIME': 'TEMPORAL_RELATION',
            'MONEY': 'MONETARY_RELATION',
            'PERCENT': 'QUANTITATIVE_RELATION',
            'CARDINAL': 'QUANTITATIVE_RELATION',
            'ORDINAL': 'QUANTITATIVE_RELATION'
        }
        
        for label in spacy_labels:
            if label in spacy_to_relation_map:
                relation = spacy_to_relation_map[label]
                self.add_relation(relation)
    
    def __len__(self):
        return len(self.rel_to_id)
    
    def __contains__(self, relation: str):
        return relation in self.rel_to_id
    
    def __getitem__(self, relation: str):
        return self.get_id(relation)


# 全局关系类型管理器实例
_global_relation_manager = None

def get_global_relation_manager() -> RelationTypeManager:
    """获取全局关系类型管理器实例"""
    global _global_relation_manager
    if _global_relation_manager is None:
        vocab_path = "checkpoints/relation_vocab.json"
        _global_relation_manager = RelationTypeManager(vocab_path)
    return _global_relation_manager

def save_global_relation_vocab():
    """保存全局关系类型词汇表"""
    manager = get_global_relation_manager()
    manager.save_vocab("checkpoints/relation_vocab.json")
