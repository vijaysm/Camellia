# Phatch - Photo Batch Processor
# Copyright (C) 2007-2008 www.stani.be
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/
#
# Phatch recommends SPE (http://pythonide.stani.be) for editing python files.

# Embedded icon is taken from www.openclipart.org (public domain)

# Follows PEP8

from core import models
from lib.reverse_translation import _t

#---PIL


def init():
    global Image, ImageOps, imtools
    import Image
    import ImageOps
    from lib import imtools


def posterize(image, bits, amount=100):
    """Apply a filter
    - amount: 0-1"""
    image = imtools.convert_safe_mode(image)
    posterized = imtools.remove_alpha(image)
    posterized = ImageOps.posterize(posterized, bits)
    if imtools.has_alpha(image):
        imtools.put_alpha(posterized, imtools.get_alpha(image))
    if amount < 100:
        return imtools.blend(image, posterized, amount / 100.0)
    return posterized

#---Phatch


class Action(models.Action):
    """"""

    label = _t('Posterize')
    author = 'Stani'
    email = 'spe.stani.be@gmail.com'
    init = staticmethod(init)
    pil = staticmethod(posterize)
    version = '0.1'
    tags = [_t('color')]
    __doc__ = _t('Reduce the number of bits of color channel')

    def interface(self, fields):
        fields[_t('Bits')] = self.SliderField(1, 0, 8)
        fields[_t('Amount')] = self.SliderField(100, 1, 100)

    icon = \
'x\xda\x01\x10\t\xef\xf6\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x000\x00\
\x00\x000\x08\x06\x00\x00\x00W\x02\xf9\x87\x00\x00\x00\x04sBIT\x08\x08\x08\
\x08|\x08d\x88\x00\x00\x08\xc7IDATh\x81\xbd\x9a{pT\xd5\x1d\xc7?\xf7\xb9\xbbw\
7\xbb\xd9<6\t\x81l\x88 \xf2P\x08\x91\x02\x82\x10@\xad\x80\xe8\xd4\xc1\xd7\
\xb4\x11\xad\xad\x1d\x15[[u:\xb6(Z\x87\x81\xe9LAm\xb1\xd0i\xa7C\xe9\xb4uD\
\x8am\xb1>F\xb4\xbc\x1c\x1f\x8d1$\x84\xa2-\x1a\x92\x182&,\xd9<\xf6yO\xff\xc8\
k\xb3l\xc2\xbd\xd8\xed\xef\x9f\xbd\xf7w\xcf=\xe7\xfb\xbd\xbf\xdf\xf9\x9d\xdf\
\xf9\x9d\x95\xb0(B\x88\xf3t\xdb\xb7o\xcf\x97e\xf9f!D\xa5i\x9a\xb9B\x88\\!\
\x84\xd74M\xafi\x9a9B\x88\x1c!\x84\x0b\x08I\x92\xd4.IR\x1b\xd0*\xcbr3\xf0\
\xa9\xaa\xaa\'\xd7\xaf_\xffAz\xbf\x92$Y\x85\x85\xe5\x96\xa9\x046m\xda4\xcb4\
\xcd\xdf\xc6\xe3\xf1J!\x842\xa8N\xaa\xaa\xda\xab\xaaj\xaf\xc3\xe1\xe8\xf7\
\xfb\xfd\xa6a\x18\x12 G\xa3Q\r\xd0\xda\xda\xdaD8\x1c\xd6#\x91\x88W\x08\xa1\n\
!\xd0u\xfd\x0b\x97\xcb\xb5\x7f\xea\xd4\xa9\xf7\xd6\xd4\xd4\xc4\xec\x12P-\xb7\
\x1c\x94\x1d;v\\\xd3\xda\xda\xfa\x92\xa2(n\xbf\xdf\xdf6y\xf2\xe4\xe8\x94)SJ\
\xbd^\xafKQ\x14\xaf\xaa\xaa\xde\xbe\xbe>:;;\t\x87\xc3\x84\xc3aJJJ0M\x93\xe2\
\xe2b\xfa\xfb\xfb\x994iRGCC\xc3\xe9\xa6\xa6&w[[[Y"\x91X\x17\x0c\x06\xa3\xc0w\
\xec\xe2\xb1M\xe0h\xe3\x9fV\xf4v\xe4\'\'\x94Lh6\x0c\xa3<\x14\nIuuuh\x9a\x86\
\xaa\xaa\x9c8q\x82\xb2\xb22|>\x1f.\x97\x8b\x96\x96\x16\x1a\x1a\x1a8~\xfc8\
\x8b\x17/f\xd1\xa2E\xc4b\xb1\xc0\x8c\x193\x02\xb3g\xcfnq8\x1c\x1fvuuU\x04+\
\x8a\xfa\xecb\x01\x9b.$I\x92|\xf5\x1d\x1c6\xdc\xc6L\xad\xeb\xcaP\xacG\xf7\
\xbb\xdd9\t]\xd7{\x14E\x11\x92$\xc9\xea\x88h\x9a\xa6\xc9.\x97\xabO\xd3\xb4\
\x84\xcf\xe7\x8b\xbb\xddn\x0c\xc3P|>\x9f\x16\x0c\x06\xf5\xa2\xe2@\xb2\xcfl=\
\xf9I\xffn\xbd\xa9\xe7\x97\x08\xc4\xfcG\xe6\n\x915\x17\xaa\xbc\x9eo\xab\x0e\
\x16\xa0\xf5IR\xf0\xa0\xd7\xe3\x82d\xd8w\xae\xbf/_\xd3d#\xae`\xc4\x05F,\x89\
\x0b\x87\xe1O\xe6\xfa\x0bM\x7f\x9e\x8b\xc2R\xbd\xc7(\xe8\x89\xaa\xbe\xcf\x12\
1\xf5\xb4\x16J|j\xb4$N\xe7\xc6\xce\x84\xca\x80\xc2\x94!\x1e\x04\x9e\xb3\x83\
\xc92\xd5\x9c|\nf,\xe1\x98\xe1\xa3Xw\x81\xee\x04\xdd\x05\xda\xe0o\xaan\x94\
\xde\t\xaan\x19O\x04\x98\xf7h\x15\rV_\x90\xad6\x9cv\x15\xcf\xa9:\xc5\xa9\xba\
Q\x91U\x8c\xa1\xb7\'N`\xb3\x9d\x17,\x13H\xc4\xa8\x1e\xba\x16b\x14\xde\xb1\
\xc1_\x1c\x91Yv\x1a[&\xc0\xc0\xd79\xff\xeb\xfe\xcf\xc0K(\x92\x0e`\xday\xcb2\
\x01I\xc2\x1c\x0f\xe0\x97u\xa7%%?e\xf5\x84\x97\xc1&\x01\xcbQH\x92H\x8e\xa0b\
\xd4\xe5\x97\x05\xefP|\xcc\xce\x7f\x00Mu\xe0n\x9f(C\x8bUX6\\h\x88\xc0x\xa0.\
\x02\xbc\x84L\x91Z\xcd\xc1C\xff \x12\x89\xe0\xe9Z\x95k\x19\x13\xf6\\(\x99\
\x0ep,\x90V\xe7B\xd0{-w]v\x02\xd19\x95E\x0b\xaf\xe6\xe8;\x07\xd1\x9a\x03\x1e\
\xab\x98\xe0b,\x90\x0e0M\xac\x82W$\x9d\x859\xcf\x129\xeb\xc2}*\x0f\xc3ps\xc5\
\xb4iD\xea\x8fY_5\xb0g\x81DF\x80i\xf3!\xf3\xcd\xa8\x9e\x98\xe9\xff\x06\xf7\
\x04\x8f\x90k\x94\xf2\xf3M?\xa6\xban\x0b\xc7>\xaa\x03\xcdEA\xc7\x11[\xf1\xcb\
\xd6$\x1eo\xb2Z\x99\xc8\x01\xa3\x92\x1b\x8bv\xe2<~\x10m\xd7Z\x94\'?e\xcb\xf3\
\xbb\xe8;\xf5O\xf6mx\x90\xda\x96\x1e\xfe\xb0\xd8\xcc\x0e\x01\xc6\x8aB\x16\
\xddiJ\xcejV\x15nE\xdfy\x0b\xb4\xd5\x83$\xb1\xffo\x7fe\xc9\xd2jr&Wq\xa7\xbf\
\x89\xbb\x9d\x9d(\x92\xd7\x16\x01\xdb.4\xee\xc25\xc6\xb3\xaa\xdc\xfbX\x9d\
\xff3\xf4\xe7\xae\x1f\x00?\xd8\xe0\xb3m7\xf3\xfcS\xdf\xa7\xa3\xa3\x83H\x7f?\
\x13=`\x8aD\x96\\\x88\x14\x0bd\x02\x9f\xa2\x1f\xb9\x95X^\xf04U\xc6\rh\xdb\
\xaa\xa1\xbb}T\x9f\xf7W\xb9\x88n\xdc\xca\x91\xc3\x87\x98\xa3\x0el\x07\xec\
\x12\xb0\x13\x85\x12\x17\x02\x9f.\xd5y\x8f3=9\x1f\xf5\x99\xea\xf3\xc0\x03\
\xc4#\xbd\xbc\xfa\xfa\x1b\xbc\xb4\xf5Q\xf2\x9c\x03\xba\xacY`\x88\xc0xI\xdc\
\x101!\xa0\xc8Y\xc9\x14s\x0e\xca\xafnB\x98}\x0c\xedQ\xc4\xb5?"i\x9a\xa8onA\
\x93L\x9a\x9f]\xcb\xc6KS\xfa\x91\xed\xe5\xb2\xb6\xe6\x80\xa5\x85\x8b\x81\x18\
\xbf<\xefq\xd4\x17\xbe\x8b\x88\xa6\xec\x14s\x02\x88U?!\xbc\xf0{ \r\x0c\xfd\
\xe0l\x08\xb8F\x9a\x98\x8a\x9a\x9dd\x0eF\\(UD\x06+,\xf0\xde\x87\xeb\x9d=\x10\
\x1a\xcci4\'b\xceZz\xa7\xae\xe4\xc3\xdaZ\xfc\x85\xc5\xf4-\xdf0\x06"%k\x16\
\x88\x0f\xa0L\x01\x9c\xf2|\x88H\xae^A\x85q\rR\xed\x0b\xc3\xcf\xa2_\xb9\x97\
\xd6y?\xe0\xe4Y\x89\xf6\x13\x1f\x10\n\x85\xe8\xbd\xf2[\x19\xc71U\xcd:z\xecY \
>V\x98L\xbd\x9e./A\xae\xdd\x03\xe6H\xeew\xf8x\x0b\xa53\x17\x10\xfe\xf8]&\xe4\
:\xf9\xbc\xa5\x99\xc7~\xf8H\xc6A\x84\xa2f)\n1\x92J\x881\xac\x80\x80\x80Q\x05\
\xf5\xfbF\xb5\x9b\xecI I\x12\xa5W^\x87\xee\x0b\x90\x97_\xc0\xda\x9e\x173\x13\
P\xb34\x07\x86](u\xb0\xb4p\xaa\xc8:N\xb5\x00\x119\x97\xaa\xc61\xefV\x9a\x9b\
\x9b\x91+o\xa1\xee\xed\xfd<to\r\xd7M\x1a\xe3C\xfb\xcb\xadB\x02lZ`\x18\xf0\
\x18\xa9t\x91:\x03\xb9\xeb\xf4\xf0\xfdP\x93X\xfe\xa5|R\xff>\xe5\xb3\xafbF\
\xeb\xcbl\xf0\x1d@\xceT\x0fqz\t\xcf\xbf\xa1\xdf\x06&[\x04b\x19\xb5)DrbN\xe8\
\xef\x1e\xad\xf6\x97s.\x14"\xd7\xed \x99L\xe2\x980\x8d\x99\xf9\x19\xfaQ\x1dt\
U\\\xc2\x9f\xe3O[.\xa9\x80\xbd\x958\x9e\x9e\xef\xa4G\xa4\x90\xf99\xb8|\xc3\
\xd61\x17\xddO\xe2\xa1wh\xae?\xc2\x9c\xa5\xab8z\xf8 g>z\xeb\xfc\xbe\x0b.\x81\
{\xf6\xf2V\xb0\x89$\x89,\xad\x03bd\x0e\x881\x12\xb7N\xb9\x1da\xf8\x87\xf5]\
\xd3oARu\xbcr\x94\xee\xeen\xf6m\x7f\x92e\x13\xd3\xfau\xf9\xe0\x9b{hm\xd8\xca\
\xa9@\x04\xb2U\x95`p\x12\xa7\xfb~\xea}\\D\x07\xe2\xf8\xac5\x08O\x11\xc2\xe9\
\xa5\xf6\xfdw1\x8a&s\xdb\xfc2\xeev\x1c\x1a\xfd\xb2\xb7\x04\xeez\x81\xceC\x8f\
\xb1\xa7\xfc\xcd!m\xd6V\xe2h\xfa\xc2\x95)\xa9;\xd9\xf6k\x92_\xddHo\xf9R\n\n\
\x8bH\x9cm\xa5\x17\'\xaf]\x1f\xe6\xf2!\xdf\x97\x15\x98W\x03\x0f\xbfGg\xdd3\
\xfc\xbe\xe2U\xe2jzO\xd6\xc4z2\x97\xeeB\xa9\xc3\xa5Z\xc1\xb7\x90\xee\xeen\
\x1a\x92\xd3\x99{\xea\x143\x17,\xe3\x95\xdd\xdbY1q.\xe4\x04`\xear\x98\xb9\
\x1a\xb3q\x1f\xc7\xfeR\xc5\xdb\x97u\xa4\x82\x87l\xd5\x85\x18\x8cB\xe3VU\x04\
\x04\\\x95\x1c8\xf8w\xae(~\x8f\x8f\xdewP:\xf1\x0e$\xc5\x01W?@\xe2\xdc)z\xcf}\
H\xe3\x81\xa7\xa9\xad\x08\x13\xc9\\D\xcc\x1e\x81\xb1\xca\x8a\xa9\xee\x14\x8b\
&\xf9\xda\x9a\xdbq8k\xd8\xb1\xff\xeb\xec\xae\xd9\xcdm\xeb\x9b\xf8\x05\x10)dt\
1=\xb3d-\x95\x88Zi\xb4\xef\x95]466\xa2\xc8\x1a\x85\x81\x00+\x1ek"\x12\x84\
\x88\xf5bIv, \xc4\xe0Bv\xa1\xb2b\xc5\x1b\x1c\xaeW).)\xe2L{;EA;p\xd2G\xb8\xb0\
\xd8!\x10\xb5\xb2\xa1\xcf/\x85\x95ew\x02\x10s7^L\x89=\xcb\x16\xe0\xc2\xa5\
\x94\xbd{_\xe4_\'O\xb0\xeca;P\x86%K\x04\xcc\x819`\xa5\xb8\x15)\xdd\xcb\xe5sF\
?\xb3!\xd9q!O\x1e\xf5\xfdab\x80\x9e>\xccy{\x82\xf2\x81\xf31\xb0\r\x1e\xe0\
\xa8\x9d\xc6\x96\xa3\xd0\x9b\xbf\x11\r\xc1Y\xdc\xa49\x88\\Le\xce\xa2\xbc\x86\
\xcdSJ;a\x94?>!^\x9dp)78\xdc\xf4\xc0\xf8\xe5u\x9b\xc7M&\xb0\x13\xb8\xfd\x91\
\xb9\xf6h_\xd4\x7f%\x9e: y[\xde\xae\xfc]\xc7\x17\xad+\x14_\x87GO;jM=bM=vM\
\x17\xd5\xccK\xe4\xf7\xdd\xf8o\x87(\xdav\xeb\xd2-;\x87Ae\xfb\xcf\x1eC\xf2\
\xc4\xc6\r\x9b\xff\xf3YS\r9\xedqW\xd9\xc7\x9a\xe29["\xa9\tY\x1f\x83\x80d:M%^\
x\xc6\xd3{\xd5\xe7\x9e\x9ej\x87&\xf2N;4c\xdb\x9a5k^O\xed\xff\xffF`H6o\xde\
\xbc2\x14\n\xddn\x9a\xf1"\xc5\x15\xf3\xa8\x0e\x91\xd4t3\xa69TS\xc5\x13Pe\xa7\
\xd0UW\xbb&{>\xd6Tg\x9d\xaa\xaauB\x88\xe3\xeb\xd6\xad\xcb\xb8\xba\xdb!\xf0_\
\xb5#\xc6\x15\x08y\xb7 \x00\x00\x00\x00IEND\xaeB`\x82\xc8\xf6o1'