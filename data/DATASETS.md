## How to write a fast enough dataset.
Nvidia's DALI library is hugely faster than any other option because
it allows GPU accelerated jpeg decoding with NVImageCodec. so that being
said you need to use jpeg images as data.


## The ~eternal~ external source
the external source is a callable python object, so either a function, or a
class implementing the `__call__(self, *args)` method.
