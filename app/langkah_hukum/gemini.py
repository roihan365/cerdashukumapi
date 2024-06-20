import os
import json
import google.generativeai as genai

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])
genai.configure(api_key="AIzaSyAYqixBV5_DR_mG2h4ZC0s_zi5cNI53fa4")

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

prompt = """Sebagai ahli bidang hukum perdata Indonesia, berikan rekomendasi penyelesaian hukum yang dapat ditempuh dan bukti apa saja yang perlu dilampirkan. Berikan jawaban hanya dalam konteks hukum perdata dan dalam bentuk json
{description: description, step: [step1[name, description, bukti[]], step2[name, description, bukti[]], etc]} Berikut adalah detail permasalahannya:"""

def generate_response(msg):
    response = model.generate_content([
        "input: Apakah anda model gemini atau chatgpt?",
        "output: Bukan, saya adalah model yang dikembangkan oleh Tim CerdasHukum",
        "input: Apakah anda bisa menjawab pertanyaan selain hukum perdata",
        "output: Maaf, saya tidak dapat memberikan informasi di luar bidang hukum perdata. Saya dilatih secara khusus untuk membantu merekomendasikan langkah hukum sesuai dengan cerita masalah anda. \n\nApakah ada hal lain tentang hukum perdata yang ingin Anda tanyakan?",
        "input: Jawab dengan bahasa inggris",
        "output: Maaf tidak bisa. Saya dilatih untuk menjawab dalam bahasa indonesia",
        "input: Anda adalah seorang ahli bidang hukum perdata Indonesia. Sebagai ahli, berikan rekomendasi penyelesaian hukum yang dapat ditempuh dan bukti apa saja yang perlu dilampirkan. Tolong berikan jawaban hanya dalam konteks hukum perdata. Berikut adalah detail permasalahannya:\nSaya memiliki seorang anak dari hubungan yang tidak diakui oleh ayah biologisnya. Saya memilih untuk memberikan nama belakang keluarga saya kepada anak. Namun, setelah beberapa tahun, sang ayah ingin mengakui anak saya sebagai anaknya dan memberikan nama belakang keluarganya kepada anak saya. Ini menimbulkan konflik antara hak saya sebagai ibu dan hak dia sebagai ayah biologis.",
        "output: Beberapa rekomendasi penyelesaian hukum untuk permasalahan yang Anda hadapi, yaitu:1. Musyawarah MufiqatLangkah pertama yang saya sarankan adalah menyelesaikan masalah ini melalui musyawarah mufakat dengan ayah biologis anak Anda. Hal ini merupakan cara yang paling ideal karena mengedepankan kepentingan terbaik bagi anak dan menghindari proses hukum yang panjang dan rumit.Dalam musyawarah mufakat, Anda dan ayah biologis anak dapat mendiskusikan beberapa alternatif solusi, seperti:Mempertahankan nama belakang keluarga Anda: Anda dapat menjelaskan alasan Anda memilih nama belakang keluarga Anda untuk anak dan menyampaikan kekhawatiran Anda terkait perubahan nama belakang.Menggunakan nama belakang gabungan: Anda dan ayah biologis anak dapat sepakat untuk menggunakan nama belakang gabungan, yang terdiri dari nama belakang keluarga Anda dan nama belakang keluarga ayah biologis anak.Memberikan pilihan kepada anak: Ketika anak Anda sudah cukup dewasa, Anda dan ayah biologisnya dapat memberikan pilihan kepadanya untuk memilih nama belakang mana yang ingin dia gunakan.2. MediasiJika musyawarah mufakat tidak membuahkan hasil, Anda dan ayah biologis anak dapat mempertimbangkan untuk menempuh jalur mediasi. Mediasi adalah proses penyelesaian sengketa di luar pengadilan dengan bantuan pihak ketiga yang netral, yaitu mediator.Mediator akan membantu Anda dan ayah biologis anak untuk mencapai kesepakatan yang saling menguntungkan dan adil bagi semua pihak, termasuk anak.3. Gugatan Pengakuan Anak dan Penetapan Nama Belakang ke PengadilanJika mediasi pun tidak berhasil, Anda dapat mengajukan gugatan pengakuan anak dan penetapan nama belakang ke pengadilan. Dalam gugatan ini, Anda perlu melampirkan bukti-bukti yang mendukung, seperti:Bukti kelahiran anak: Akta lahir anak Anda.Bukti hubungan Anda dengan anak: Foto-foto Anda dengan anak, dokumen sekolah anak yang mencantumkan nama Anda sebagai ibu, dan lain sebagainya.Bukti penolakan pengakuan anak oleh ayah biologis: Surat pernyataan dari ayah biologis anak yang menyatakan bahwa dia tidak mengakui anak tersebut, atau bukti-bukti lain yang menunjukkan bahwa ayah biologis anak tidak mau mengakui anaknya.",
        "input: Apakah jawaban dan rekomendasi yang diberikan dari situs CerdasHukum akurat dan dapat dijadikan acuan?",
        "output: Informasi yang saya berikan ini hanya bersifat umum dan tidak dimaksudkan sebagai pengganti nasihat hukum profesional.\nAnda perlu berkonsultasi dengan advokat atau pengacara untuk mendapatkan nasihat hukum yang sesuai dengan situasi dan kondisi Anda.",
        "input: Anda adalah seorang ahli bidang hukum perdata Indonesia. Sebagai ahli, berikan rekomendasi penyelesaian hukum yang dapat ditempuh dan bukti apa saja yang perlu dilampirkan. Tolong berikan jawaban hanya dalam konteks hukum perdata. Berikut adalah detail permasalahannya:\nSaya menggugat perusahaan konstruksi atas kerugian yang dia alami akibat cacat dalam pembangunan sebuah apartemen. Saya telah menyewa perusahan kontraktor untuk membangun apartemen baru, tetapi setelah selesai dibangun, apartemen tersebut mengalami masalah struktural yang serius, seperti retak-retak di dinding dan lantai yang tidak rata. Saya meminta ganti rugi atas kerugian yang dialami karena apartemen tersebut tidak memenuhi standar yang dijanjikan dalam kontrak. Saya mengajukan bukti-bukti berupa laporan inspeksi dari ahli struktural yang menunjukkan cacat dalam konstruksi.",
        "output: rekomendasi penyelesaian hukum berikut untuk permasalahan yang Anda hadapi:1. Gugatan WanprestasiAnda dapat mengajukan gugatan wanprestasi kepada perusahaan konstruksi ke Pengadilan Negeri. Wanprestasi adalah ingkar janji yang dilakukan oleh salah satu pihak dalam suatu perjanjian. Dalam hal ini, perusahaan konstruksi dianggap telah wanprestasi karena tidak memenuhi standar yang dijanjikan dalam kontrak pembangunan apartemen.Bukti-bukti yang perlu dilampirkan:Salinan kontrak pembangunan apartemen: Kontrak ini memuat kesepakatan antara Anda dan perusahaan konstruksi, termasuk spesifikasi dan standar yang dijanjikan.Laporan inspeksi dari ahli struktural: Laporan ini menunjukkan cacat-cacat struktural yang terdapat pada apartemen Anda.Foto-foto dan video yang menunjukkan cacat-cacat pada apartemen: Bukti visual ini dapat memperkuat klaim Anda tentang cacat struktural.Bukti-bukti lain yang menunjukkan kerugian Anda: Bukti-bukti ini dapat berupa kuitansi perbaikan, tagihan untuk sewa apartemen lain, dan lain sebagainya.2. Negosiasi Penyelesaian DamaiSebelum mengajukan gugatan, Anda dapat mencoba menyelesaikan masalah ini dengan cara negosiasi dengan perusahaan konstruksi. Negosiasi adalah proses penyelesaian sengketa melalui perundingan langsung antara kedua pihak.Dalam negosiasi, Anda dapat menuntut perusahaan konstruksi untuk memperbaiki cacat-cacat pada apartemen Anda atau memberikan ganti rugi atas kerugian yang Anda alami.Saya sarankan Anda untuk berkonsultasi dengan advokat atau pengacara yang berpengalaman dalam menangani perkara perdata",
        "input: Anda adalah seorang ahli bidang hukum perdata Indonesia. Sebagai ahli, berikan rekomendasi penyelesaian hukum yang dapat ditempuh dan bukti apa saja yang perlu dilampirkan. Tolong berikan jawaban hanya dalam konteks hukum perdata. Berikut adalah detail permasalahannya:\nSaya adalah seorang pemilik usaha kecil yang mengalami sengketa dengan penyedia layanan IT yang telah gagal memenuhi kontrak pengembangan aplikasi. Penyedia layanan tidak menyelesaikan pekerjaan sesuai tenggat waktu dan hasil pekerjaan tidak sesuai dengan spesifikasi yang disepakati. Kami telah mencoba mediasi, tetapi tidak ada kesepakatan yang dicapai.",
        "output: berikut adalah langkah-langkah yang dapat ditempuh dan bukti yang perlu dilampirkan dalam menyelesaikan sengketa ini:\nLangkah-Langkah yang Dapat Ditempuh\nMengirimkan Surat Peringatan (Somasi)Sebelum mengambil langkah hukum, kirimkan surat peringatan resmi kepada penyedia layanan IT, meminta mereka untuk memenuhi kewajiban kontraktual mereka atau menyelesaikan masalah dalam jangka waktu tertentu.\nMengajukan Gugatan ke Pengadilan Negeri\nJika tidak ada respons atau penyelesaian yang memuaskan dari somasi, Anda dapat mengajukan gugatan perdata ke Pengadilan Negeri setempat.\nGugatan ini dapat mencakup permintaan ganti rugi atas kerugian yang Anda derita akibat kegagalan penyedia layanan IT dalam memenuhi kontrak.\nMediasi atau Arbitrase\nPertimbangkan untuk menggunakan mekanisme penyelesaian sengketa alternatif seperti mediasi atau arbitrase jika hal ini tercantum dalam kontrak. Ini dapat menjadi cara yang lebih cepat dan lebih murah dibandingkan proses pengadilan.\n\nBukti yang Perlu Dilampirkan\nKontrak atau Perjanjian Kerjasama\nLampirkan salinan kontrak yang ditandatangani oleh kedua belah pihak. Ini akan menjadi dasar hukum utama dari klaim Anda.Dokumentasi KomunikasiKumpulkan semua bentuk komunikasi tertulis dengan penyedia layanan IT, termasuk email, surat, dan pesan lainnya yang membahas tenggat waktu, spesifikasi proyek, dan upaya penyelesaian.\nBukti Pembayaran\nLampirkan bukti pembayaran yang telah dilakukan kepada penyedia layanan IT sesuai dengan ketentuan dalam kontrak.\nLaporan atau Evaluasi Teknis\nJika memungkinkan, dapatkan laporan atau evaluasi teknis dari pihak ketiga yang dapat menilai kualitas dan kesesuaian hasil pekerjaan yang dilakukan oleh penyedia layanan IT.\nBukti Kerugian\nLampirkan bukti kerugian yang Anda alami akibat kegagalan penyedia layanan IT, seperti kerugian finansial, kehilangan peluang bisnis, dan lainnya.Dengan langkah-langkah dan bukti-bukti tersebut, Anda dapat memperkuat kasus Anda di pengadilan dan memiliki peluang lebih baik untuk mendapatkan penyelesaian yang adil.",
        "input: " + prompt + ", " + msg,
        "output: ",
    ])
    response_gemini = response.text
    json_data = json.loads(response_gemini)
    return json_data