MACHINE LEARNING PRACTICAL EXAM PACK (10 PRACTICALS)
===================================================

Folder structure:
- practical_01.py ... practical_10.py : separate programs
- datasets/ : separate CSV datasets and sample media
- requirements.txt : Python libraries
- offline_setup_windows.bat : helper to install from a local wheelhouse
- download_wheels_online.bat : run this on an internet-connected Windows PC
- outputs/ : generated results will be saved here

RECOMMENDED PREPARATION
1. Install 64-bit Python 3.11 on your practice computer (and on the exam PC if allowed).
2. Open Command Prompt in this folder.
3. Install dependencies online with: python -m pip install -r requirements.txt
4. Run any practical: python practical_01.py (change number as needed).

OFFLINE EXAM PREPARATION
A. On an internet-connected Windows PC with the SAME Python version/architecture as the exam PC:
   - Double-click download_wheels_online.bat
   - It downloads wheels and dependencies into wheelhouse/
   - Copy the whole ML_Practical_Exam_Pack folder to a USB drive.
B. On the offline exam PC:
   - Copy the folder from USB to the PC.
   - Open Command Prompt in this folder.
   - Run: python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
   - Then run a practical, e.g. python practical_01.py

IMPORTANT:
- Wheel files must match the exam PC's OS, Python version and architecture. Windows 64-bit Python 3.11 is the intended setup.
- If Python itself is not installed on the exam PC, a wheelhouse is not enough: ask the lab/faculty whether Python is preinstalled or whether you may carry the official Python installer.
- Practical 9 uses a bundled generated sample CCTV-style video and OpenCV background subtraction; no pretrained model or internet is needed.
- Practical 10 uses a bundled small multimodal demonstration dataset and local sample images. It is an educational toy dataset, not a reliable real-world fake-news detector.
- Results/plots are saved to outputs/. Some programs also display a plot window.
