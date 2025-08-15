# row_column_attention: Use @Qwen
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config, Qwen3RotaryEmbedding

class RowColumnAttention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.row_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=0)
        self.column_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=1)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self._weights_loaded = False

    def load_weights_once(self, row_weights: Dict, col_weights: Dict):
        self.row_attention_layer.load_state_dict(row_weights)
        self.column_attention_layer.load_state_dict(col_weights)
        self._weights_loaded = True

    def forward(
        self,
        embedded_rows: List[torch.Tensor],
        row_attention_mask: Optional[List[torch.Tensor]] = None,
        column_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        row_layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        col_layer_weights: Optional[Dict[str, torch.Tensor]] = None
    ):

        if not self._weights_loaded:
            if row_layer_weights is None:
                raise ValueError("Qwen3DecoderLayer weights should be provided for row_attention_layer in param 'row_layer_weights'")
            if col_layer_weights is None:
                raise ValueError("Qwen3DecoderLayer weights should be provided for column_attention_layer in param 'col_layer_weights'")
            self.load_weights_once(row_layer_weights, col_layer_weights)

        if row_attention_mask is None:
            row_attention_mask = [None] * len(embedded_rows)
        
        
        processed_rows = []
        container_tokens = []

        for row_embed, row_mask in zip(embedded_rows, row_attention_mask):
            if row_embed.dim() == 2:
                row_input = row_embed.unsqueeze(0) # shape -> [1, seq_len, hidden_size]
            else:
                row_input = row_embed

            seq_len = row_input.shape[1]
            position_ids = torch.arange(seq_len, device=row_input.device).unsqueeze(0)

            if position_embeddings is None:
                pos_emb = self.rotary_emb(row_input, position_ids)
            else:
                pos_emb = position_embeddings
            # pos_emb = self.rotary_emb(row_input, position_ids)

            row_output = self.row_attention_layer(
                hidden_states=row_input,
                attention_mask=row_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb
            )

            row_output = row_output[0] # QwenDecoderLayer returns a tuple, (hidden_state, attention_weight)
            processed_row = row_output.squeeze(0) # shape -> [seq_len, hidden_size]
            processed_rows.append(processed_row)

            container_token = processed_row[0:1] # shape -> [1, hidden_size]
            container_tokens.append(container_token)
        
        container_tensor = torch.cat(container_tokens, dim=0)
        container_input = container_tensor.unsqueeze(0) # shape -> [1, num_rows, hidden_size]
        num_rows = container_input.shape[1]
        col_position_ids = torch.arange(num_rows, device=container_input.device).unsqueeze(0)

        if position_embeddings is None:
            col_pos_emb = self.rotary_emb(container_input, col_position_ids)
        else:
            col_pos_emb = position_embeddings
        # col_pos_emb = self.rotary_emb(container_input, col_position_ids)

        col_output = self.column_attention_layer(
            hidden_states=container_input,
            attention_mask=column_attention_mask,
            position_ids=col_position_ids,
            position_embeddings=col_pos_emb
        )
        col_output = col_output[0]
        final_containers = col_output.squeeze(0) # shape -> [num_rows, hidden_size]

        updated_rows = []
        for i, original_row in enumerate(processed_rows):
            updated_row = original_row.clone()
            updated_row[0] = final_containers[i]
            updated_rows.append(updated_row)
        
        return updated_rows

if __name__ == "__main__":
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import torch
    
    print("🔍 开始测试 RowColumnAttention...")
    
    try:
        # 1. 从HuggingFace加载Qwen3配置
        print("📥 从HuggingFace加载Qwen3配置...")
        # 尝试加载Qwen3相关配置
        try:
            # 如果有Qwen3模型，可以直接加载
            config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")  # 或其他Qwen3模型
            print("✅ 成功加载Qwen3配置")
        except:
            # 如果没有，创建一个典型的Qwen3配置
            print("⚠️  未找到Qwen3-0.6B，创建默认配置")
            config = Qwen3Config(
                vocab_size=151936,
                hidden_size=1024,
                num_attention_heads=16,
                num_key_value_heads=8,
                num_hidden_layers=24,
                intermediate_size=2048,
                max_position_embeddings=32768
            )
        
        print(f"   - Model: Qwen3")
        print(f"   - Hidden Size: {config.hidden_size}")
        print(f"   - Attention Heads: {config.num_attention_heads}")
        print(f"   - Layers: {config.num_hidden_layers}")
        
        # 2. 创建处理器
        processor = RowColumnAttention(config)
        print("✅ RowColumnAttention 创建成功")
        
        # 3. 准备测试数据
        embedded_rows = [
            torch.randn(5, config.hidden_size),  # 行1：5个token
            torch.randn(3, config.hidden_size),  # 行2：3个token
            torch.randn(7, config.hidden_size),  # 行3：7个token
        ]
        
        print("✅ 测试数据准备完成")
        print(f"   - 行数: {len(embedded_rows)}")
        for i, row in enumerate(embedded_rows):
            print(f"   - 行{i}形状: {row.shape}")
        
        # 4. 加载真实的Qwen3权重
        print("📥 加载Qwen3权重...")
        try:
            # 尝试加载真实的Qwen3模型
            qwen_model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
            
            # 获取前两层的权重
            row_weights = qwen_model.model.layers[0].state_dict()
            col_weights = qwen_model.model.layers[1].state_dict()
            
            print("✅ 成功加载Qwen3权重")
            print(f"   - 行权重参数数: {len(row_weights)}")
            print(f"   - 列权重参数数: {len(col_weights)}")
            
        except Exception as e:
            print(f"⚠️  无法加载真实Qwen3权重: {e}")
            print("💡 使用模拟权重进行测试")
            # 创建模拟权重
            row_weights = {name: torch.randn_like(param) for name, param in processor.row_attention_layer.named_parameters()}
            col_weights = {name: torch.randn_like(param) for name, param in processor.column_attention_layer.named_parameters()}
        
        # 5. 测试前向传播
        print("🔄 开始前向传播测试...")
        
        try:
            output_rows = processor(
                embedded_rows=embedded_rows,
                row_layer_weights=row_weights,
                col_layer_weights=col_weights
            )
            
            print("✅ 第一次前向传播成功!")
            print(f"   - 输出行数: {len(output_rows)}")
            for i, row in enumerate(output_rows):
                print(f"   - 输出行{i}形状: {row.shape}")
            
            # 6. 测试第二次调用
            print("🔄 测试第二次调用...")
            output_rows_2 = processor(embedded_rows=embedded_rows)
            print("✅ 第二次调用成功!")
            
            print("\n🎉 所有测试通过!")
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()