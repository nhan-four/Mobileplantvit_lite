"""
Script để xóa cache validation của dataset.
Chạy script này khi bạn muốn force re-validate tất cả ảnh.
"""

import argparse
from pathlib import Path
import shutil


def clear_cache(data_root: str, verbose: bool = True):
    """
    Xóa cache validation cho một dataset.
    
    Args:
        data_root: Đường dẫn đến thư mục dataset (chứa train/val/test)
        verbose: In thông tin chi tiết
    """
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Không tìm thấy thư mục: {data_root}")
        return False
    
    # Tìm cache directory
    cache_dirs = []
    
    # Cache ở cấp dataset root
    cache_dir = data_path / ".validation_cache"
    if cache_dir.exists():
        cache_dirs.append(cache_dir)
    
    # Cache ở các thư mục con (train/val/test)
    for subdir in ["train", "val", "test"]:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            sub_cache_dir = subdir_path.parent / ".validation_cache"
            if sub_cache_dir.exists() and sub_cache_dir not in cache_dirs:
                cache_dirs.append(sub_cache_dir)
    
    if not cache_dirs:
        if verbose:
            print(f"ℹ️  Không tìm thấy cache nào trong {data_root}")
        return True
    
    # Xóa cache
    deleted_count = 0
    for cache_dir in cache_dirs:
        try:
            if verbose:
                cache_files = list(cache_dir.glob("*.json"))
                print(f"🗑️  Đang xóa {len(cache_files)} file cache trong {cache_dir}")
            
            shutil.rmtree(cache_dir)
            deleted_count += 1
            
            if verbose:
                print(f"✓ Đã xóa {cache_dir}")
        except Exception as e:
            print(f"❌ Lỗi khi xóa {cache_dir}: {e}")
    
    if verbose:
        print(f"\n✓ Đã xóa {deleted_count} cache directory")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Xóa cache validation của dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Xóa cache cho một dataset cụ thể
  python clear_validation_cache.py --data-root /path/to/dataset
  
  # Xóa cache cho tất cả CCMT datasets
  python clear_validation_cache.py --all-ccmt
  
  # Xóa cache mà không in thông tin chi tiết
  python clear_validation_cache.py --data-root /path/to/dataset --quiet
        """
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        help="Đường dẫn đến thư mục dataset (chứa train/val/test)"
    )
    
    parser.add_argument(
        "--all-ccmt",
        action="store_true",
        help="Xóa cache cho tất cả CCMT datasets (Cashew, Cassava, Maize, Tomato)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Không in thông tin chi tiết"
    )
    
    args = parser.parse_args()
    
    if args.all_ccmt:
        # Xóa cache cho tất cả CCMT datasets
        base_path = Path("/home/nhannv02/Hello/plantvit_lite/dataset/Dataset_for_Crop_Pest_and_Disease_Detection/Data_split")
        crops = ["Cashew", "Cassava", "Maize", "Tomato"]
        seeds = [42, 123, 999]
        
        print("🗑️  Xóa cache cho tất cả CCMT datasets...")
        print("="*80)
        
        total_cleared = 0
        for crop in crops:
            for seed in seeds:
                data_root = base_path / crop / f"seed_{seed}"
                if data_root.exists():
                    if not args.quiet:
                        print(f"\n📁 {crop} - seed_{seed}")
                    if clear_cache(str(data_root), verbose=not args.quiet):
                        total_cleared += 1
        
        print("\n" + "="*80)
        print(f"✓ Đã xóa cache cho {total_cleared} datasets")
        
    elif args.data_root:
        # Xóa cache cho một dataset cụ thể
        clear_cache(args.data_root, verbose=not args.quiet)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

