#!/bin/bash
# 전체 pipeline 자동화:
# 1. 압축 검증 → 원본 CSV 삭제 (31GB 확보)
# 2. resume_parallel.py 실행 (남은 70% Goldsky 데이터)
# 3. iter14c 재실행 → 새 calibration
# 4. 모든 자매 프로젝트 diverse strategies 재실행
# 5. dashboard 갱신

set -e
cd c:/Projects/Quant/polymarket-analyze

echo "=========================================="
echo " STAGE 1: 압축 검증 + 원본 삭제"
echo "=========================================="
cd data/poly_data
xz -t orderFilled_complete.csv.xz && echo "  ✅ 무결성 OK" || { echo "❌ 압축 손상"; exit 1; }
rm orderFilled_complete.csv
echo "  원본 삭제 완료"
df -h /c | head -3

echo ""
echo "=========================================="
echo " STAGE 2: resume_parallel (남은 데이터)"
echo "=========================================="
PYTHONIOENCODING=utf-8 python -u resume_parallel.py --workers 4

echo ""
echo "=========================================="
echo " STAGE 3: iter14c calibration (full data)"
echo "=========================================="
cd ..
PYTHONIOENCODING=utf-8 python -u iters/iter14b_calibration_v2.py

echo ""
echo "=========================================="
echo " STAGE 4: 자매 프로젝트 diverse strategies"
echo "=========================================="
cd c:/Projects/Quant/EquityDistribution
for i in $(seq 31 35); do
  PYTHONIOENCODING=utf-8 python iters/iter_diverse_strategies.py --round $i 2>&1 | tail -2
done

cd c:/Projects/Quant/OptionsDistribution
for i in $(seq 31 35); do
  PYTHONIOENCODING=utf-8 python iters/iter_diverse_strategies.py --round $i 2>&1 | tail -2
done

cd c:/Projects/Quant/ProbabilityDistribution
for i in $(seq 31 35); do
  PYTHONIOENCODING=utf-8 python iter_diverse_strategies.py --round $i 2>&1 | tail -2
done

echo ""
echo "=========================================="
echo " STAGE 5: dashboard 갱신"
echo "=========================================="
cd c:/Projects/Quant/iter_queue
PYTHONIOENCODING=utf-8 python update_dashboards.py

echo ""
echo " ✅ 전체 PIPELINE 완료"
