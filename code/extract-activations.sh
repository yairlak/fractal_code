# Launch extract-activations.py

for depth in 1 2 3; do
	for length in 1 2 3; do
		python extract-activations.py --model ../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt --vocab ../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt.vocab --p1 0.5 --p2 0.5 --depth $depth --length $length --path2output ../results/activations/
	done
done
