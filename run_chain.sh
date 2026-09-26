#!/bin/bash
# Recovery chain after the Colab runtime was recycled: restore -> forward-k4 rebuild (reference 0.98662) -> exact rank-fix rerun -> test-pool dense encoding.
# Each stage backs exp/ + logs up to Drive.  Sequential on purpose (GPU shared between XGBoost and the encoder caused an OOM before).
until grep -q RESTORED /content/restore.log 2>/dev/null; do sleep 20; done
cd /content/ber
cp /content/exp_*.py src/
B=/content/drive/MyDrive/ber_ckpt
mkdir -p $B/logs
bk() { rsync -a exp/ $B/exp/; cp /content/*.log $B/logs/ 2>/dev/null; echo "BK $1 $(date -u +%FT%TZ)" >> $B/logs/chain_status.txt; }
export PYTHONIOENCODING=utf-8
python3 -u src/exp_fwd.py > /content/fwd.log 2>&1; echo "exp_fwd rc=$?" >> /content/chain.log; bk fwd
python3 -u src/exp_rankfix.py > /content/rankfix2.log 2>&1; echo "exp_rankfix rc=$?" >> /content/chain.log; bk rankfix
(cd src && python3 -u -c "import exp_dense as d; d.build('test')" > /content/dense_test.log 2>&1); echo "dense_test rc=$?" >> /content/chain.log; bk dense
echo CHAIN_DONE >> /content/chain.log
