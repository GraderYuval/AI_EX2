206217457
204755516
*****
Comments:
better_evaluation_function description:
1- we keep our higher tiles in the upper-left part of the board by punishing high tiles in bottom row and right column, and by rewarding high tiles in top left corner.
2- we keep our board ordered in a monotonous order, from bottom right to top left, by calculating the difference between adjacent cells.
3- we reward a large quantity of zero-tiles, to encourage an empty board
4- we punish for large differences between adjacent tiles, to avoid small numbers interrupting between larger tiles
5- we add a constant to avoid negative numbers
