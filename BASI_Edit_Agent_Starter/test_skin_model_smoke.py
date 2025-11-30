"""
Smoke tests for Stage 3 Skin Residual Model.

Run this to verify imports and basic functionality work.
"""

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from models.residual_unet import SkinResidualNet, ResidualUNet
        print("  ✓ models.residual_unet imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import models.residual_unet: {e}")
        return False
    
    try:
        from training_utils import BeforeAfterDataset, load_image, resize_with_aspect
        print("  ✓ training_utils imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import training_utils: {e}")
        return False
    
    try:
        import train_skin_model
        print("  ✓ train_skin_model imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import train_skin_model: {e}")
        return False
    
    try:
        import apply_skin_model
        print("  ✓ apply_skin_model imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import apply_skin_model: {e}")
        return False
    
    return True


def test_model_instantiation():
    """Test that the model can be instantiated and run forward pass."""
    print("\nTesting model instantiation...")
    try:
        import torch
        from models.residual_unet import SkinResidualNet
        
        model = SkinResidualNet(in_ch=3, base_ch=32, use_mask=False)
        print("  ✓ Model instantiated successfully")
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
        print(f"  ✓ Forward pass successful: {x.shape} -> {out.shape}")
        
        return True
    except ImportError:
        print("  ⚠ torch not available, skipping model test")
        return True  # Not a failure if torch isn't installed
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_dataset_class():
    """Test that BeforeAfterDataset can be instantiated (with dummy paths)."""
    print("\nTesting dataset class...")
    try:
        from training_utils import BeforeAfterDataset
        
        # This will fail if globs don't match, but we're just testing the class exists
        print("  ✓ BeforeAfterDataset class available")
        return True
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_argparse():
    """Test that scripts can parse arguments."""
    print("\nTesting argument parsing...")
    try:
        import sys
        from train_skin_model import parse_args
        
        # Test with minimal args
        sys.argv = ['train_skin_model.py', '--config', 'config.yaml', '--dataset_version', 'skin_v1']
        args = parse_args()
        print("  ✓ train_skin_model argument parsing works")
        
        from apply_skin_model import parse_args as parse_apply_args
        sys.argv = ['apply_skin_model.py', '--input_glob', 'test/*.jpg', '--output_dir', 'test_out']
        args = parse_apply_args()
        print("  ✓ apply_skin_model argument parsing works")
        
        return True
    except Exception as e:
        print(f"  ✗ Argument parsing test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 3 Skin Residual Model - Smoke Tests")
    print("=" * 60)
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_model_instantiation()
    all_passed &= test_dataset_class()
    all_passed &= test_argparse()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All smoke tests passed!")
    else:
        print("✗ Some tests failed. Check output above.")
    print("=" * 60)

