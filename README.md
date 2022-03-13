# IF3230_Tubes2

1. Jelaskan cara kerja program Anda, terutama pada paralelisasi dengan CUDA yang Anda implementasikan berdasarkan skema di atas.

2. Dari waktu eksekusi terbaik program paralel Anda, bandingkan dengan waktu eksekusi program sekuensial yang diberikan. Analisis mengapa waktu eksekusi program Anda bisa lebih lambat / lebih cepat / sama saja. Lalu simpulkan bagaimana CUDA memengaruhi waktu eksekusi program Anda. Buktikan dengan menunjukkan waktu eksekusi yang diperlukan saat demo.

3. Jelaskan secara singkat apakah ada perbedaan antara hasil keluaran program serial dan program paralel Anda, dan jika ada jelaskan juga penyebab dari perbedaan tersebut.

4. Dengan paralelisasi yang Anda implementasikan, untuk bagian perhitungan konvolusi saja, dari 3 kasus berikut yang manakah yang waktu eksekusinya paling cepat dan mengapa?

    Jumlah Matrix: 10000, Ukuran Kernel: 1x1, Ukuran Matrix: 1x1

    Jumlah Matrix: 1, Ukuran Kernel: 1x1, Ukuran Matrix: 100x100

    Jumlah Matrix: 1, Ukuran Kernel: 100x100, Ukuran Matrix: 100x100

    (Note: ketiga kasus memiliki jumlah operasi perkalian yang sama)
