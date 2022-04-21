"""
    File regrouping general unittests.
"""

import os
        
def test_src_access():
    """
        Is src folder accessible from the test folder.
    """
    assert 'src' in os.listdir()

        
        
