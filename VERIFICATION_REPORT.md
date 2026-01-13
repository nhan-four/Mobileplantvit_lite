# BÁO CÁO XÁC MINH CẤU HÌNH BASELINE
## MobilePlantViT-Lite++: Linear Attention + MLP FFN

**Ngày kiểm tra:** 2025-01-03  
**Cấu hình kiểm tra:**
- `attn_type=linear`
- `ffn_type=mlp`
- `embed_dim=256`
- `encoder_depth=1`
- `ffn_expand=2.0`

---

## 1. XÁC MINH MODULE SWITCHING ✅ PASS

### 1.1. Attention Type
- **Yêu cầu:** `attn_type=linear` phải khởi tạo `LinearSelfAttention`
- **Kết quả:** ✅ **PASS**
- **Chi tiết:**
  - Module được khởi tạo: `LinearSelfAttention`
  - Không có low-rank matrices (`kv_down`, `kv_up`)
  - Có `qkv` projection đầy đủ: `embed_dim × (1 + 2×embed_dim)`
  - Có `out` projection: `embed_dim × embed_dim`

**Code location:** `mobileplantvit/models/encoder.py:174-175`
```python
elif attn_type == "linear":
    self.attn = LinearSelfAttention(embed_dim)
```

**Logic điều kiện:** ✅ Đúng - dựa trên `attn_type`, không phụ thuộc vào `attn_rank > 0`

### 1.2. FFN Type
- **Yêu cầu:** `ffn_type=mlp` phải khởi tạo `MLPFFN` (2-layer MLP)
- **Kết quả:** ✅ **PASS**
- **Chi tiết:**
  - Module được khởi tạo: `MLPFFN`
  - Không có Conv layers (`nn.Conv2d`)
  - Có 2 Linear layers: `fc1` và `fc2`
  - Cấu trúc: `Linear → GELU → Dropout → Linear → Dropout`

**Code location:** `mobileplantvit/models/encoder.py:183-184`
```python
elif ffn_type == "mlp":
    self.ffn = MLPFFN(embed_dim, expand_ratio=ffn_expand, dropout=dropout)
```

**Logic điều kiện:** ✅ Đúng - dựa trên `ffn_type`, không phụ thuộc vào `ffn_kernel` hoặc default expand settings

### 1.3. Không có Module Reuse
- **Kiểm tra:** Đảm bảo không có shared/cached modules từ factorized/tokenconv path
- **Kết quả:** ✅ **PASS**
- **Chi tiết:**
  - Mỗi `EncoderBlockLitePP` khởi tạo modules riêng biệt
  - Không có module sharing giữa các blocks
  - Logic điều kiện rõ ràng, không có fallback về factorized/tokenconv

---

## 2. KIỂM TRA SỐ LƯỢNG THAM SỐ ✅ PASS

### 2.1. Tính toán dự kiến

**Per Encoder Block:**

| Component | Calculation | Params |
|-----------|-------------|--------|
| **LinearSelfAttention** | | |
| - qkv weight | `256 × (1 + 512) = 256 × 513` | 131,328 |
| - qkv bias | `1 + 512 = 513` | 513 |
| - out weight | `256 × 256` | 65,536 |
| - out bias | `256` | 256 |
| **Attention Total** | | **197,633** |
| **LayerNorm (2x)** | `256 × 2 × 2` (weight + bias) | **1,024** |
| **MLPFFN** | | |
| - fc1 weight | `256 × 512` | 131,072 |
| - fc1 bias | `512` | 512 |
| - fc2 weight | `512 × 256` | 131,072 |
| - fc2 bias | `256` | 256 |
| **FFN Total** | | **262,912** |
| **Block Total** | | **461,569** |

**Total encoder_layers (1 block):** 461,569 params

### 2.2. Số lượng tham số thực tế

- **encoder_layers params:** 461,569 ✅
- **Chênh lệch với tính toán:** 0 (100% chính xác) ✅

### 2.3. So sánh với Lite++ Reference

- **Lite++ reference (factorized + tokenconv):** 386,113 params
- **Baseline (linear + mlp):** 461,569 params
- **Tăng:** +75,456 params (+19.54%) ✅
- **Khoảng dự kiến:** 70,000 - 100,000 params
- **Kết quả:** ✅ **PASS** - Nằm trong khoảng dự kiến

### 2.4. Tổng số tham số model

- **Total params:** 586,030 (0.586M) ✅
- **Yêu cầu:** >= 580,000
- **Kết quả:** ✅ **PASS**

---

## 3. KIỂM TRA LOGIC ĐIỀU KIỆN ✅ PASS

### 3.1. Attention Path

**File:** `mobileplantvit/models/encoder.py:168-177`

```python
attn_type = (attn_type or "factorized").lower().strip()
if attn_type == "factorized":
    self.attn = FactorizedLinearSelfAttention(...)
elif attn_type == "linear":
    self.attn = LinearSelfAttention(embed_dim)
else:
    raise ValueError(...)
```

**Đánh giá:**
- ✅ Logic điều kiện dựa trên `attn_type`
- ✅ Không phụ thuộc vào `attn_rank > 0`
- ✅ Không có fallback hoặc default về factorized

### 3.2. FFN Path

**File:** `mobileplantvit/models/encoder.py:179-186`

```python
ffn_type = (ffn_type or "tokenconv").lower().strip()
if ffn_type == "tokenconv":
    self.ffn = TokenConvFFN(...)
elif ffn_type == "mlp":
    self.ffn = MLPFFN(embed_dim, expand_ratio=ffn_expand, dropout=dropout)
else:
    raise ValueError(...)
```

**Đánh giá:**
- ✅ Logic điều kiện dựa trên `ffn_type`
- ✅ Không phụ thuộc vào `ffn_kernel` hoặc default expand settings
- ✅ `ffn_expand` được áp dụng đúng cho MLPFFN

### 3.3. Không có Side Effects

- ✅ `attn_rank` chỉ được sử dụng khi `attn_type="factorized"`
- ✅ `ffn_kernel` chỉ được sử dụng khi `ffn_type="tokenconv"`
- ✅ Không có shared state hoặc cached modules

---

## 4. CẤU TRÚC MODULE

```
EncoderBlockLitePP (461,569 params)
├── LayerNorm (512 params)
├── LayerNorm (512 params)
├── LinearSelfAttention (197,633 params)
│   ├── Linear (qkv: 131,841 params)
│   └── Linear (out: 65,792 params)
└── MLPFFN (262,912 params)
    ├── Linear (fc1: 131,584 params)
    ├── GELU (0 params)
    ├── Dropout (0 params)
    └── Linear (fc2: 131,328 params)
```

**Xác nhận:**
- ✅ Không có Conv layers trong FFN
- ✅ Không có low-rank matrices trong attention
- ✅ Cấu trúc đúng với yêu cầu

---

## 5. KẾT LUẬN

### ✅ PASS - Tất cả các kiểm tra đều PASS

**Tóm tắt:**
1. ✅ Module switching hoạt động đúng: `attn_type=linear` → `LinearSelfAttention`, `ffn_type=mlp` → `MLPFFN`
2. ✅ Logic điều kiện đúng: dựa trên `attn_type` và `ffn_type`, không phụ thuộc vào các tham số khác
3. ✅ Số lượng tham số chính xác: 461,569 params cho encoder_layers, tăng 75,456 so với Lite++ reference
4. ✅ Tổng params: 586,030 >= 580,000 (yêu cầu)
5. ✅ Không có module reuse hoặc side effects

**Cấu hình baseline (Linear + MLP) là công bằng và đúng đắn, phù hợp để sử dụng làm reference cho ablation study.**

---

## 6. LƯU Ý QUAN TRỌNG

⚠️ **Cảnh báo:** File `run_config.json` trong thư mục `runs_mobileplantvit/20260103_184834_linear+mlp/` cho thấy model đã được train với:
- `attn_type: "factorized"` ❌
- `ffn_type: "tokenconv"` ❌

**Đây KHÔNG phải là cấu hình baseline!** Model này sử dụng factorized attention và tokenconv FFN, giống với Lite++ main model.

Để train với cấu hình baseline đúng, cần chạy:
```bash
python train.py --model litepp \
  --encoder_depth 1 \
  --embed_dim 256 \
  --ffn_expand 2.0 \
  --head_type gap \
  --attn_type linear \
  --ffn_type mlp \
  --main_data_root <path>
```

---

**Báo cáo được tạo bởi:** `verify_baseline_config.py`  
**Ngày:** 2025-01-03

