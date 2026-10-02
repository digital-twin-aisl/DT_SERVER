# VGGT-Omega 패치

`vggt-omega-patch-tokens.patch`는 VGGT-Omega 서브모듈(커밋 `39a0cb8`)에
`Aggregator.encode_patch_tokens` / `forward_patch_tokens`와
`VGGTOmega.forward_patch_tokens`를 추가합니다. 엣지가 계산한 DINO patch token으로
서버에서 보정을 이어 가는 분산 보정(`calibration_worker/inference.py`)에 필요합니다.

```bash
git submodule update --init apps/calibration_worker/vggt-omega
git -C apps/calibration_worker/vggt-omega apply ../patches/vggt-omega-patch-tokens.patch
```

**라이선스**: 이 패치는 VGGT-Omega 코드의 수정본이므로 intelligent-synchronization의 LGPL이 아니라
**FAIR Noncommercial Research License**(`LICENSES/LicenseRef-FAIR-Noncommercial-Research.txt`)를
따릅니다. 비상업 연구 목적에만 사용할 수 있습니다.
