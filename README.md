# Audio Air Studio

Ứng dụng desktop để **upsample 24 → 48 kHz** và **EQ độ thoáng (air)** cho giọng nói, có xem waveform/phổ tần và nghe thử trước khi xuất.

## Chạy

- Nhấp đôi **`Chay_AudioAir.bat`** (tự dùng bản .exe nếu có), hoặc
- Chạy trực tiếp **`dist\Audio Air Studio\Audio Air Studio.exe`**, hoặc
- `python AudioAirStudio.py` (có thể truyền sẵn đường dẫn file: `python AudioAirStudio.py audio.wav`)

## Tính năng

| Nhóm | Chi tiết |
|---|---|
| Mở file | WAV, FLAC, MP3, OGG, AIFF; kéo-thả file vào cửa sổ; Ctrl+O |
| Hiển thị | Waveform toàn file (vùng nghe thử tô màu cam) + phổ tần trước/sau xử lý |
| Preset | Nguyên bản, Tự nhiên +1 dB, Air +2 dB, Sáng +3 dB @ 7 kHz, Chi tiết +4 dB @ 10 kHz, Giọng nữ, Giọng nam, Tùy chỉnh |
| EQ | High-shelf: mức nâng −6…+6 dB, tần số 4–14 kHz, Q 0.4–1.6 (zero-phase `sosfiltfilt`) |
| Nghe thử | Nghe gốc, nghe sau xử lý, **A/B 5 giây** (gốc ↔ xử lý), thanh âm lượng |
| Xuất | WAV 16/24/32-bit float, 44.1/48/96 kHz hoặc giữ nguyên, chuẩn hóa đỉnh −1 dBFS |
| A/B file | Xuất bản 5 giây đổi gốc/xử lý để kiểm tra trong Premiere/DaVinci |

## Phím tắt

| Phím | Chức năng |
|---|---|
| `Ctrl+O` | Mở file |
| `Space` | Nghe bản sau xử lý |
| `P` | Nghe bản gốc |
| `Esc` | Dừng phát |
| `Ctrl+E` | Xuất WAV |
| Bấm waveform | Đặt điểm bắt đầu đoạn nghe thử |

## Ghi chú kỹ thuật

- Resample bằng `scipy.signal.resample_poly` (polyphase 2x), không dùng FFT resample nên không bị ringing đầu/cuối file.
- Nguồn 24 kHz chỉ có nội dung thật tới 12 kHz; upsample không tạo thêm thông tin, nhưng giữ chất lượng khi dựng/xuất lại trong Premiere và tránh bị resample thêm lần nữa.
- High-shelf chỉ nâng vùng trên tần số đã chọn, không đụng phần thân giọng; mức 1–3 dB là đủ, cao hơn dễ chói âm "s".
- File nghe thử tạm được ghi vào `%TEMP%\audioair_*` và tự xóa khi đóng app.

## Đóng gói lại thành .exe

Dùng môi trường sạch để tránh PyInstaller kéo theo toàn bộ site-packages (có thể lên vài GB):

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install pyinstaller numpy scipy matplotlib PyQt5 soundfile
.\.venv-build\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed ^
  --name "Audio Air Studio" AudioAirStudio.py
```

Bản đóng gói nằm trong `dist\Audio Air Studio\` (~235 MB). Kiểm tra nhanh bản đóng gói không cần mở cửa sổ:

```powershell
"dist\Audio Air Studio\Audio Air Studio.exe" --self-test audio.wav
Get-Content "$env:TEMP\audioair_selftest.txt"
```
