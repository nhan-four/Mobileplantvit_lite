"""
Script test để kiểm tra logic validation đã được tối ưu hóa.
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from mobileplantvit.data.datasets import AlbumentationsImageFolder


def test_validation(data_root: str, num_runs: int = 2):
    """
    Test validation logic với và không có cache.
    
    Args:
        data_root: Đường dẫn đến thư mục dataset
        num_runs: Số lần chạy để test cache
    """
    print("="*80)
    print("TEST VALIDATION LOGIC")
    print("="*80)
    
    # Clear cache trước khi test
    cache_dir = Path(data_root).parent / ".validation_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ Đã xóa cache cũ: {cache_dir}\n")
    
    for run in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{num_runs}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Test với cache enabled
        dataset = AlbumentationsImageFolder(
            root=data_root,
            transform=None,
            return_path=False,
            enable_cache=True,
            num_workers=4
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Thời gian: {elapsed:.2f}s")
        print(f"📊 Kết quả:")
        print(f"   - Tổng ảnh: {len(dataset.base)}")
        print(f"   - Ảnh hợp lệ: {len(dataset.valid_indices)}")
        print(f"   - Ảnh lỗi: {len(dataset.corrupted_files)}")
        
        if run == 0:
            print(f"\n💡 Lần chạy đầu tiên: validation từ đầu")
        else:
            print(f"\n💡 Lần chạy thứ {run + 1}: sử dụng cache (nhanh hơn)")
    
    print("\n" + "="*80)
    print("✓ TEST HOÀN TẤT")
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test validation logic")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/nhannv02/Hello/plantvit_lite/dataset/Dataset_for_Crop_Pest_and_Disease_Detection/Data_split/Cashew/seed_42/train",
        help="Đường dẫn đến thư mục dataset để test"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Số lần chạy để test cache"
    )
    
    args = parser.parse_args()
    
    if not Path(args.data_root).exists():
        print(f"❌ Không tìm thấy thư mục: {args.data_root}")
        return
    
    test_validation(args.data_root, args.runs)


if __name__ == "__main__":
    main()

