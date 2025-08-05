"""
Test suite for fpyjp.utils.list_utils module.

This test suite provides comprehensive coverage for all functions in the list_utils module,
including edge cases, error conditions, and various parameter combinations.
"""

import pytest

from fpyjp.utils.list_utils import ensure_list, pad_list, get_padded_value_at_period


class TestPadArray:
    """Test cases for the pad_list function."""
    
    def test_basic_padding_zero_mode(self):
        """Test basic padding with zero mode (default)."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 5)
        assert result == [1.0, 2.0, 3.0, 0.0, 0.0]
    
    def test_basic_padding_last_mode(self):
        """Test basic padding with last mode."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 5, pad_mode='last')
        assert result == [1.0, 2.0, 3.0, 3.0, 3.0]
    
    def test_padding_with_positive_start(self):
        """Test padding with positive start position."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 6, start=2)
        assert result == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]
    
    def test_padding_with_positive_start_last_mode(self):
        """Test padding with positive start and last mode."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 6, start=2, pad_mode='last')
        assert result == [0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    
    def test_padding_with_negative_start(self):
        """Test padding with negative start position."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 5, start=-3)
        assert result == [0.0, 0.0, 1.0, 2.0, 3.0]
    
    def test_padding_with_negative_start_equal_to_length(self):
        """Test padding with start = -len(values) (special case)."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 5, start=-3)  # start == -len(values)
        assert result == [0.0, 0.0, 1.0, 2.0, 3.0]
    
    def test_padding_with_very_negative_start(self):
        """Test padding with very negative start (should be clamped to 0)."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 5, start=-10)
        assert result == [1.0, 2.0, 3.0, 0.0, 0.0]
    
    def test_empty_input_array(self):
        """Test padding with empty input array."""
        result = pad_list([], 3)
        assert result == [0.0, 0.0, 0.0]
    
    def test_zero_target_length(self):
        """Test padding with zero target length."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 0)
        assert result == []
    
    def test_negative_target_length(self):
        """Test padding with negative target length."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, -1)
        assert result == []
    
    def test_input_longer_than_target(self):
        """Test when input array is longer than target length."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pad_list(values, 3)
        assert result == [1.0, 2.0, 3.0]
    
    def test_input_longer_than_target_with_start(self):
        """Test when input array is longer than target length with start offset."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = pad_list(values, 4, start=2)
        assert result == [0.0, 0.0, 1.0, 2.0]
    
    def test_single_element_array(self):
        """Test padding with single element array."""
        values = [5.0]
        result = pad_list(values, 4, pad_mode='last')
        assert result == [5.0, 5.0, 5.0, 5.0]
    
    def test_exact_fit(self):
        """Test when input array exactly fits target length."""
        values = [1.0, 2.0, 3.0]
        result = pad_list(values, 3)
        assert result == [1.0, 2.0, 3.0]


class TestGetPaddedValueAtPeriod:
    """Test cases for the get_padded_value_at_period function."""
    
    def test_basic_value_access(self):
        """Test basic value access within array bounds."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 0) == 1.0
        assert get_padded_value_at_period(values, 1) == 2.0
        assert get_padded_value_at_period(values, 2) == 3.0
    
    def test_out_of_bounds_zero_mode(self):
        """Test out of bounds access with zero padding mode."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 3) == 0.0
        assert get_padded_value_at_period(values, 10) == 0.0
    
    def test_out_of_bounds_last_mode(self):
        """Test out of bounds access with last padding mode."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 3, pad_mode='last') == 3.0
        assert get_padded_value_at_period(values, 10, pad_mode='last') == 3.0
    
    def test_with_positive_start(self):
        """Test value access with positive start offset."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 0, start=2) == 0.0
        assert get_padded_value_at_period(values, 1, start=2) == 0.0
        assert get_padded_value_at_period(values, 2, start=2) == 1.0
        assert get_padded_value_at_period(values, 3, start=2) == 2.0
        assert get_padded_value_at_period(values, 4, start=2) == 3.0
        assert get_padded_value_at_period(values, 5, start=2) == 0.0
    
    def test_with_positive_start_last_mode(self):
        """Test value access with positive start offset and last mode."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 5, start=2, pad_mode='last') == 3.0
    
    def test_with_negative_start(self):
        """Test value access with negative start offset."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 2, n_length=5, start=-3) == 1.0
        assert get_padded_value_at_period(values, 3, n_length=5, start=-3) == 2.0
        assert get_padded_value_at_period(values, 4, n_length=5, start=-3) == 3.0
        assert get_padded_value_at_period(values, 0, n_length=5, start=-3) == 0.0
        assert get_padded_value_at_period(values, 1, n_length=5, start=-3) == 0.0
    
    def test_with_negative_start_equal_to_length(self):
        """Test negative start equal to array length (special case)."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 2, n_length=5, start=-3) == 1.0
        assert get_padded_value_at_period(values, 3, n_length=5, start=-3) == 2.0
        assert get_padded_value_at_period(values, 4, n_length=5, start=-3) == 3.0
    
    def test_negative_start_without_n_length_raises_error(self):
        """Test that negative start without n_length raises ValueError."""
        values = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="n_length is required when start < 0"):
            get_padded_value_at_period(values, 0, start=-1)
    
    def test_empty_array(self):
        """Test with empty array."""
        assert get_padded_value_at_period([], 0) == 0.0
        assert get_padded_value_at_period([], 5) == 0.0
    
    def test_negative_period(self):
        """Test with negative period."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, -1) == 0.0
        assert get_padded_value_at_period(values, -5) == 0.0
    
    def test_period_beyond_n_length(self):
        """Test period beyond specified n_length."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 10, n_length=5, start=0) == 0.0
    
    def test_zero_n_length(self):
        """Test with zero n_length."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 0, n_length=0, start=-1) == 0.0
    
    def test_negative_n_length(self):
        """Test with negative n_length."""
        values = [1.0, 2.0, 3.0]
        assert get_padded_value_at_period(values, 0, n_length=-1, start=-1) == 0.0
    
    def test_single_element_array(self):
        """Test with single element array."""
        values = [5.0]
        assert get_padded_value_at_period(values, 0) == 5.0
        assert get_padded_value_at_period(values, 1, pad_mode='last') == 5.0
        assert get_padded_value_at_period(values, 1, pad_mode='zero') == 0.0


class TestEnsureList:
    """Test cases for the ensure_list function."""
    
    def test_scalar_int(self):
        """Test conversion of scalar integer."""
        result = ensure_list(5)
        assert result == [5.0]
        assert isinstance(result[0], float)
    
    def test_scalar_float(self):
        """Test conversion of scalar float."""
        result = ensure_list(3.14)
        assert result == [3.14]
        assert isinstance(result[0], float)
    
    def test_list_of_ints(self):
        """Test conversion of list of integers."""
        result = ensure_list([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)
    
    def test_list_of_floats(self):
        """Test conversion of list of floats."""
        result = ensure_list([1.5, 2.5, 3.5])
        assert result == [1.5, 2.5, 3.5]
        assert all(isinstance(x, float) for x in result)
    
    def test_mixed_list(self):
        """Test conversion of mixed integer/float list."""
        result = ensure_list([1, 2.5, 3])
        assert result == [1.0, 2.5, 3.0]
        assert all(isinstance(x, float) for x in result)
    
    def test_empty_list(self):
        """Test conversion of empty list."""
        result = ensure_list([])
        assert result == []
    
    def test_zero_value(self):
        """Test conversion of zero values."""
        assert ensure_list(0) == [0.0]
        assert ensure_list(0.0) == [0.0]
        assert ensure_list([0, 0.0]) == [0.0, 0.0]
    
    def test_negative_values(self):
        """Test conversion of negative values."""
        assert ensure_list(-5) == [-5.0]
        assert ensure_list([-1, -2.5]) == [-1.0, -2.5]


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_pad_list_and_get_value_consistency(self):
        """Test that pad_list and get_padded_value_at_period are consistent."""
        values = [1.0, 2.0, 3.0]
        n_length = 6
        start = 2
        
        # Create padded array
        padded = pad_list(values, n_length, start=start, pad_mode='zero')
        
        # Check consistency for each position
        for period in range(n_length):
            expected = padded[period]
            actual = get_padded_value_at_period(values, period, n_length=n_length, start=start, pad_mode='zero')
            assert actual == expected, f"Mismatch at period {period}: expected {expected}, got {actual}"
    
    def test_pad_list_and_get_value_consistency_last_mode(self):
        """Test consistency with last padding mode."""
        values = [1.0, 2.0, 3.0]
        n_length = 6
        start = 1
        
        # Create padded array
        padded = pad_list(values, n_length, start=start, pad_mode='last')
        
        # Check consistency for each position
        for period in range(n_length):
            expected = padded[period]
            actual = get_padded_value_at_period(values, period, n_length=n_length, start=start, pad_mode='last')
            assert actual == expected, f"Mismatch at period {period}: expected {expected}, got {actual}"
    
    def test_ensure_list_with_other_functions(self):
        """Test ensure_list output works with other functions."""
        # Test with scalar input
        values = ensure_list(5.0)
        result = pad_list(values, 3, pad_mode='last')
        assert result == [5.0, 5.0, 5.0]
        
        # Test with list input
        values = ensure_list([1, 2, 3])
        assert get_padded_value_at_period(values, 1) == 2.0