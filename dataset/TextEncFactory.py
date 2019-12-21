from skipthoughts import BayesianUniSkip, UniSkip, BiSkip, DropUniSkip

def get_text_enc(config, vocab):
    skipthoughts_dir, text_enc = config['skipthoughts_dir'], config['txt_enc']
    if text_enc == 'BayesianUniSkip':
        return BayesianUniSkip(skipthoughts_dir, vocab)
    if text_enc == 'UniSkip':
        return UniSkip(skipthoughts_dir, vocab)
    if text_enc == 'BiSkip':
        return BiSkip(skipthoughts_dir, vocab)
    if text_enc == 'DropUniSkip':
        return DropUniSkip(skipthoughts_dir, vocab)
        
