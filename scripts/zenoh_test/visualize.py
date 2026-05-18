# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def visualize():
    if not os.path.exists('zenoh_test_results.csv'):
        print("결과 파일(zenoh_test_results.csv)이 존재하지 않습니다. 먼저 서버 코드를 실행하여 결과를 수집하세요.")
        sys.exit(1)

    df = pd.read_csv('zenoh_test_results.csv')

    # QoS 타입별로 데이터 분할
    tensor_df = df[df['type'] == 'tensor_best_effort']
    control_df = df[df['type'] == 'control_reliable']

    # 통계 출력
    print("========== Zenoh QoS 통신 지연시간 분석 결과 ==========")
    if not tensor_df.empty:
        total_expected = 100 # 10fps * 10초
        received = len(tensor_df)
        drop_rate = ((total_expected - received) / total_expected) * 100
        print(f"[Best-Effort 텐서] 평균 지연시간: {tensor_df['latency_ms'].mean():.2f} ms")
        print(f"[Best-Effort 텐서] 최대 지연시간: {tensor_df['latency_ms'].max():.2f} ms")
        print(f"[Best-Effort 텐서] 패킷 드랍률: {drop_rate:.1f}% ({received}/{total_expected})")

    if not control_df.empty:
        print(f"[Reliable 제어] 평균 지연시간: {control_df['latency_ms'].mean():.2f} ms")
    print("=======================================================\n")

    # 그래프 시각화 설정
    plt.figure(figsize=(12, 6))

    if not tensor_df.empty:
        plt.plot(tensor_df['id'], tensor_df['latency_ms'], label='Tensor (Best-Effort: 1MB)', 
                 marker='o', linestyle='-', color='#1f77b4', alpha=0.8)

    if not control_df.empty:
        plt.plot(control_df['id'], control_df['latency_ms'], label='Control (Reliable: Bytes)', 
                 marker='s', linestyle='-', color='#d62728', markersize=8)

    plt.title('Zenoh E2E Latency: Edge to Cloud', fontsize=16, fontweight='bold')
    plt.xlabel('Message Sequence ID', fontsize=12)
    plt.ylabel('End-to-End Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 이미지 파일로 저장
    plt.savefig('zenoh_latency_chart.png', dpi=300)
    print("시각화 차트가 'zenoh_latency_chart.png' 파일로 저장되었습니다.")

if __name__ == "__main__":
    visualize()
