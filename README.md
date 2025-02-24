### Color Thief Fast

This project is dedicated to extracting image theme colors. During the development process, we referred to
[Color Thief](https://github.com/fengsp/color-thief-py/). and greatly improved the processing speed by introducing the
numpy library.

1. High performance processing
   With the powerful array computing power of numpy, the speed has been increased by more than 5 times, and compared to
   traditional methods, the performance has been significantly optimized.
2. Custom pixel processing
   Support users to customize pixel processing according to their own needs, meeting diverse image processing
   requirements

### Installation

```shell
pip install color-thief-fast
```

### Usage

```python
from color_thief_fast import ColorThiefFast

color_thief = ColorThiefFast('/path/to/imagefile', return_percent=True)
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
# build a color palette
palette = color_thief.get_palette(color_count=5)
```

### Thanks

Thanks to Lokesh Dhakar for his [original work](https://github.com/lokesh/color-thief/).  
Thanks to Shipeng Feng for his  [original work](https://github.com/fengsp/color-thief-py/).

### Better

If you feel anything wrong, feedbacks or pull requests are welcome.