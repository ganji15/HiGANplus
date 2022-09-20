ImgHeight = 64
CharWidth = ImgHeight // 2

data_roots = {
    'iam': './data/iam/'
}

data_paths = {
    'iam_word': {'trnval': 'trnvalset_words%d.hdf5'%ImgHeight,
                 'test': 'testset_words%d.hdf5'%ImgHeight},
    'iam_line': {'trnval': 'trnvalset_lines%d.hdf5'%ImgHeight,
                 'test': 'testset_lines%d.hdf5'%ImgHeight},
    'iam_word_org': {'trnval': 'trnvalset_words%d_OrgSz.hdf5'%ImgHeight,
                     'test': 'testset_words%d_OrgSz.hdf5'%ImgHeight}
}