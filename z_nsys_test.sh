nsys profile -o ./z_offset/z_offset_0  --stats=true -t cuda   ./element_add  0  > z_0.log
nsys profile -o ./z_offset/z_offset_4  --stats=true -t cuda   ./element_add  1  > z_1.log
nsys profile -o ./z_offset/z_offset_8  --stats=true -t cuda   ./element_add  2  > z_2.log
nsys profile -o ./z_offset/z_offset_12  --stats=true -t cuda  ./element_add  3  > z_3.log
nsys profile -o ./z_offset/z_offset_16  --stats=true -t cuda  ./element_add  4  > z_4.log
nsys profile -o ./z_offset/z_offset_20  --stats=true -t cuda  ./element_add  5  > z_5.log
nsys profile -o ./z_offset/z_offset_24  --stats=true -t cuda  ./element_add  6  > z_6.log
nsys profile -o ./z_offset/z_offset_28  --stats=true -t cuda  ./element_add  7  > z_7.log
nsys profile -o ./z_offset/z_offset_32  --stats=true -t cuda  ./element_add  8  > z_8.log