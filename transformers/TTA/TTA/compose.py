import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, images=None, results=None, target_sizes=None):
        for transform in self.transforms:
            images, results = transform(images, results, target_sizes)
        
        if images is not None and results is not None:
            return images, results  
        elif images is not None:
            return images
        elif results is not None:
            return results

        
def horizontal_flip(images=None, results=None, target_sizes=None):
    if images is not None:
        images = [np.fliplr(image) for image in images]

    if (results is not None) ^ (target_sizes is not None):
        raise Exception("results를 transform 시 target_sizes가 지정되어야 합니다.")

    if results is not None and target_sizes is not None:
        for result, target_size in zip(results, target_sizes):
            x1 = result['boxes'][:, 0]
            x2 = result['boxes'][:, 2]
            w, h, _ = target_size
            x1, x2 = w - x2, w - x1

            result['boxes'][:, 0] = x1
            result['boxes'][:, 2] = x2
    
    return images, results

def identity(images=None, results=None, target_sizes=None):
    return images, results
