# def _print_scores(self, words, tags, score):
#     table = [['' for j in range(len(words)+1)] for i in range(len(tags)+1)]
#     word_range = range(len(words))
#     for j in word_range:
#         table[0][j+1] = words[j]
#     for i in range(len(tags)):
#         table[i+1][0] = tags[i]
#         for j in word_range:
#             if score[i][j] != None:
#                 table[i+1][j+1] = ("%.3g" % score[i][j])
#             else:
#                 table[i+1][j+1] = '_'
#     self._print_table(table)
#             
#     
# def _print_table(self, table):
#     """Prints out a table of data, padded for alignment
#     @param out: Output stream (file-like object)
#     @param table: The table to print. A list of lists.
#     Each row must have the same number of columns. """
#     col_paddings = []
# 
#     for i in range(len(table[0])):
#         col_paddings.append(self.get_max_width(table, i))
# 
#     for row in table:
#         # left col
#         print row[0].ljust(col_paddings[0] + 1),
#         # rest of the cols
#         for i in range(1, len(row)):
#             col = row[i].rjust(col_paddings[i] + 2)
#             print col,
#         print
#         
# def get_max_width(self, table, index):
#     """Get the maximum width of the given column index"""
#     return max([len(row[index]) for row in table])