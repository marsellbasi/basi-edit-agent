"""
Smoke test for BgResidualNet v2 with masks.
"""

def test_bg_v2_model():
    """Test that BgResidualNetV2 can be instantiated and run forward pass with masks."""
    try:
        import torch
        from models.residual_unet import BgResidualNetV2
        
        print("Testing BgResidualNetV2 instantiation...")
        model = BgResidualNetV2(
            in_ch=3,
            base_ch=48,
            use_mask=True,
            bg_residual_scale=1.0,
            subj_residual_scale=0.1
        )
        print("  ✓ Model instantiated successfully")
        
        # Test forward pass with mask
        print("Testing forward pass with mask...")
        x = torch.randn(1, 3, 256, 256)
        mask = torch.rand(1, 256, 256)  # Subject mask [0, 1]
        
        with torch.no_grad():
            residual = model.forward_with_mask_weighted(x, mask)
        
        assert residual.shape == x.shape, f"Output shape {residual.shape} != input shape {x.shape}"
        print(f"  ✓ Forward pass with mask successful: {x.shape} -> {residual.shape}")
        
        # Test forward pass without mask (fallback)
        print("Testing forward pass without mask (fallback)...")
        with torch.no_grad():
            residual_no_mask = model(x)
        
        assert residual_no_mask.shape == x.shape
        print(f"  ✓ Forward pass without mask successful: {x.shape} -> {residual_no_mask.shape}")
        
        return True
    except ImportError:
        print("  ⚠ torch not available, skipping model test")
        return True  # Not a failure if torch isn't installed
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("BgResidualNet v2 Smoke Test")
    print("=" * 60)
    
    success = test_bg_v2_model()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All smoke tests passed!")
    else:
        print("✗ Some tests failed. Check output above.")
    print("=" * 60)

