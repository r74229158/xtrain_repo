import torch


class AddCustomTransformations:
    """
    A transform class that adds random_noize and other elements to the images using PyTorch operations.
    """

    def __init__(self, add_grid=False, add_lines=False, add_noize=False,
                  dot_prob=0.02, grid_spacing=7, grid_value=1, max_lines=5):
        """
        Args:
            add_grid (bool): If true, a random grid appears on images (different for each image)
            add_lines (bool): If true, random, short lines appears on images (different for each image)
            add_noize (bool): If true, random noize is added to the images

            dot_prob (float): controls the magnitude of noize addition
            max_lines (int): max number of lines to add (add_lines should be true)
        """
        
        self.flag_noize = add_noize
        self.flag_grid = add_grid
        self.flag_lines = add_lines

        self.dot_prob = dot_prob
        self.max_lines = max_lines

    def add_noize(self, noisy_img):
        
        # Add random dots (salt and pepper noise)
        pepper_mask = torch.rand_like(noisy_img) < self.dot_prob
        noisy_img[pepper_mask] = torch.maximum(noisy_img[pepper_mask], 
                                             torch.tensor(0.3))  # Light gray dots
        
        salt_mask = torch.rand_like(noisy_img) < (self.dot_prob / 2)
        noisy_img[salt_mask] = torch.maximum(noisy_img[salt_mask], 
                                           torch.tensor(0.6))  # Brighter dots
        return noisy_img
    
    def add_grid(self, noisy_img):
        """Add grid using tensor operations"""

        grid_val = torch.rand(1)/2 + 0.5
        grid_space = torch.randint(7, 13, (1,)).item()
        
        # Create vertical grid lines
        vertical_lines = torch.zeros_like(noisy_img)
        vertical_lines[:, :, ::grid_space] = grid_val
        noisy_img = torch.maximum(noisy_img, vertical_lines)
        
        # Create horizontal grid lines  
        horizontal_lines = torch.zeros_like(noisy_img)
        horizontal_lines[:, ::grid_space, :] = grid_val
        noisy_img = torch.maximum(noisy_img, horizontal_lines)
        
        return noisy_img
    
    def add_random_lines(self, noisy_img):

        c, h, w = noisy_img.shape
        
        # Determine how many lines to add (0 to max_lines)
        num_lines = torch.randint(2, self.max_lines+1, (1,)).item()
        
        for _ in range(num_lines):

            ## Randomly choose between horizontal and vertical
            is_horizontal = torch.rand(1) > 0.5
            
            ## Random line properties
            line_value = torch.rand(1).item() * 0.1 + 0.7  # Random brightness between 0.2-0.6
            line_pos = torch.randint(0, h if is_horizontal else w, (1,)).item()
            
            # line_length = torch.randint(w//3, w, (1,)).item() if is_horizontal else torch.randint(h//3, h, (1,)).item()
            line_length = torch.randint(3, 5, (1,)).item()
            
            if is_horizontal:

                # Add horizontal line
                start_pos = torch.randint(0, w - line_length + 1, (1,)).item()
                end_pos = start_pos + line_length

                # Ensure we don't go out of bounds
                end_pos = min(end_pos, w)
                
                noisy_img[:, 
                         max(0, line_pos - 1):min(h, line_pos + 1),
                         start_pos:end_pos] = torch.maximum(
                    noisy_img[:, 
                             max(0, line_pos - 1):min(h, line_pos + 1),
                             start_pos:end_pos],
                    torch.tensor(line_value)
                )
            else:

                # Add vertical line
                start_pos = torch.randint(0, h - line_length + 1, (1,)).item()
                end_pos = start_pos + line_length
                
                # Ensure we don't go out of bounds
                end_pos = min(end_pos, h)
                noisy_img[:, start_pos:end_pos, max(0, line_pos - 1):min(w, line_pos + 1)] = torch.maximum(
                    noisy_img[:, start_pos:end_pos, max(0, line_pos - 1):min(w, line_pos + 1)],
                    torch.tensor(line_value)
                )
        
        return noisy_img

    
    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: Tensor of shape (C, H, W) with values in [0, 1]
        Returns:
            Noisy tensor with same shape
        """

        # Save current state
        original_state = torch.get_rng_state()
                
        # Use the image data to create a deterministic but unique seed
        img_hash = img_tensor.sum().item()  # Simple hash based on image content
        seed = int(img_hash * 1000) % (2**32)

        torch.manual_seed(seed)

        # Work on a copy
        noisy_img = img_tensor.clone()

        if self.flag_noize:
            noisy_img = self.add_noize(noisy_img)
        
        if self.flag_grid:
            noisy_img = self.add_grid(noisy_img)
        
        if self.flag_lines:
            noisy_img = self.add_random_lines(noisy_img)
        
        # Restore original state
        torch.set_rng_state(original_state)

        return noisy_img
        
