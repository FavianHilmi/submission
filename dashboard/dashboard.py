import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Analisis Kualitas Udara", layout="wide")
st.header('Analisis Kualitas Udara di Changping dan Dongsi')

@st.cache_data
def load_data():
    # Ambil path direktori saat ini (lokasi file dashboard.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Buat path relatif ke folder data
    changping_path = os.path.join(BASE_DIR, "../data/PRSA_Data_Changping.csv")
    dongsi_path = os.path.join(BASE_DIR, "../data/PRSA_Data_Dongsi.csv")
    
    # Baca CSV dari path relatif
    changping_df = pd.read_csv(changping_path, sep=';')
    dongsi_df = pd.read_csv(dongsi_path, sep=';')
    
    changping_df['station'] = 'Changping'
    dongsi_df['station'] = 'Dongsi'
    
    df_combined = pd.concat([changping_df, dongsi_df], ignore_index=True)
    df_combined["datetime"] = pd.to_datetime(df_combined[["year", "month", "day", "hour"]])
    
    return df_combined

df = load_data()

# penanganan missing values dan Outlier
air_quality_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
selected_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
existing_columns = [col for col in selected_columns if col in df.columns]

df[existing_columns] = df[existing_columns].apply(pd.to_numeric, errors='coerce')

df[existing_columns] = df[existing_columns].fillna(df[existing_columns].mean())

# menangani outlier
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df = remove_outliers(df, existing_columns)

df.to_csv("DatasetProcessing.csv", index=False)


# Cek apakah masih ada data
# st.write(df["PM2.5"].describe(), "jumlah data PM")

# Sidebar
st.sidebar.header("Filter Data")
years = sorted(df['year'].unique())
start_year = st.sidebar.selectbox("Tahun Awal", years, index=0)
end_year = st.sidebar.selectbox("Tahun Akhir", years[::-1], index=0)
location = st.sidebar.selectbox("Pilih Lokasi", ["Changping", "Dongsi", "Semua"], index=2)
selected_param = st.sidebar.selectbox("Pilih Parameter Kualitas Udara", air_quality_columns)

start_date = datetime(start_year, 1, 1)
end_date = datetime(end_year, 12, 31)

df_filtered = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

if location != "Semua":
    df_filtered = df_filtered[df_filtered["station"] == location]

# salin dataFrame agar tidak ada warning dari Streamlit
df_filtered = df_filtered.copy()


# mapping musim
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
df_filtered["season"] = df_filtered["month"].map(season_map)
df_filtered["day_of_week"] = df_filtered["datetime"].dt.day_name()

# grouping data
avg_hourly = df_filtered.groupby("hour")[selected_param].mean().reset_index()
# avg_hourly = df_filtered.groupby("hour")[selected_param].mean()

# Cek apakah hasilnya ada
# st.write(avg_hourly)  

# Section 1 (Hourly Trends)
st.subheader(f"Tren {selected_param} Berdasarkan Jam")
# st.write("Statistik PM2.5 Setelah Pengolahan:")
# st.write(df_filtered["PM2.5"].describe())
# st.write("Filtered DataFrame:", df_filtered)
# st.write("Average Hourly Data:", avg_hourly)

chart = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": f"Rata-rata {selected_param} per Jam",
    "mark": {"type": "line", "strokeWidth": 3, "color": "#83c9ff"},
    "encoding": {
        "x": {
            "field": "hour",
            "type": "ordinal",
            "axis": {"title": "Jam", "labelColor": "#e6eaf1", "titleColor": "#e6eaf1"}
        },
        "y": {
            "field": selected_param,
            "type": "quantitative",
            "axis": {"title": selected_param, "labelColor": "#e6eaf1", "titleColor": "#e6eaf1"}
        }
    },
    "data": {"values": avg_hourly.to_dict(orient="records")},
    "config": {
        "background": "#0e1117",
        "title": {"color": "#fafafa", "fontSize": 16},
        "axis": {
            "labelFontSize": 12,
            "titleFontSize": 14,
            "gridColor": "#31333F"
        }
    }
}

st.vega_lite_chart(chart, use_container_width=True)
# Hitung jam dengan rata-rata tertinggi dan terendah
max_hour = avg_hourly.loc[avg_hourly[selected_param].idxmax(), "hour"]
min_hour = avg_hourly.loc[avg_hourly[selected_param].idxmin(), "hour"]
max_value = avg_hourly[selected_param].max()
min_value = avg_hourly[selected_param].min()

# Tampilkan kesimpulan
st.markdown(f"""
Dari analisis tren **{selected_param}** berdasarkan jam, diperoleh bahwa:  

- **Kadar {selected_param} tertinggi** terjadi pada pukul **{max_hour}:00**, dengan rata-rata **{max_value:.2f} µg/m³**.  
- **Kadar {selected_param} terendah** terjadi pada pukul **{min_hour}:00**, dengan rata-rata **{min_value:.2f} µg/m³**.  

Pola ini menunjukkan adanya **fluktuasi kadar {selected_param} sepanjang hari**, yang dapat dipengaruhi oleh aktivitas manusia, pola lalu lintas, serta faktor lingkungan seperti suhu dan kelembaban.  
""")
st.markdown("<br><hr style='height:3px;border:none;border-top:3px solid #f0f0f0;'><br>", unsafe_allow_html=True)

# section 2 (seasonal & weekly trends)
st.subheader(f"Perbedaan pola {selected_param} berdasarkan musim")

if "season" not in df_filtered.columns:
    st.error("Kolom 'season' tidak ditemukan dalam dataset!")
elif selected_param not in df_filtered.columns:
    st.error(f"Kolom {selected_param} tidak ditemukan dalam dataset!")
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df_filtered, x="season", y=selected_param, ax=ax, palette="Blues"
    )

    ax.set_xlabel("Musim")
    ax.set_ylabel(f"Kadar {selected_param} (µg/m³)")
    ax.set_title(f"Distribusi {selected_param} Berdasarkan Musim")

    st.pyplot(fig)
    
    # Hitung rata-rata per musim
    avg_per_season = df_filtered.groupby("season", observed=True)[selected_param].mean()

    # Tentukan musim dengan nilai tertinggi dan terendah
    max_season = avg_per_season.idxmax()
    min_season = avg_per_season.idxmin()
    max_value = avg_per_season.max()
    min_value = avg_per_season.min()

    # Tampilkan kesimpulan
    # st.subheader("Kesimpulan")
    st.markdown(f"""
    Dari analisis kadar **{selected_param}** berdasarkan musim, terlihat bahwa:  

    - Musim dengan **kadar {selected_param} tertinggi** adalah **{max_season}** dengan rata-rata **{max_value:.2f} µg/m³**.  
    - Musim dengan **kadar {selected_param} terendah** adalah **{min_season}** dengan rata-rata **{min_value:.2f} µg/m³**.  

    Hal ini menunjukkan adanya perbedaan signifikan dalam kadar {selected_param} di berbagai musim.  
    Faktor seperti cuaca, suhu, kelembaban, serta aktivitas manusia mungkin berkontribusi terhadap variasi ini.  
    """)
    
    # with st.expander("See explanation"):
    #     st.write(
    #         """Visualisasi ini menunjukkan kadar rata-rata dari variabel yang dipilih pada setiap musim."""
    #     )

st.markdown("<br><hr style='height:3px;border:none;border-top:3px solid #f0f0f0;'><br>", unsafe_allow_html=True)

# section 3 (daily pattern)
# st.subheader(f"Pola {selected_param} Berdasarkan Hari dalam Seminggu")
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.boxplot(x="day_of_week", y=selected_param, data=df_filtered, 
#             order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ax=ax)
# ax.set_xlabel("Hari")
# ax.set_ylabel(selected_param)
# ax.set_xticks(ax.get_xticks())
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# st.pyplot(fig)

# st.markdown("<br><hr style='height:3px;border:none;border-top:3px solid #f0f0f0;'><br>", unsafe_allow_html=True)

# Section 3 pengganti (daily pattern)
st.subheader(f"Pola {selected_param} Berdasarkan Hari dalam Seminggu")

# Perbaiki urutan hari
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Pastikan hanya data dengan hari yang valid
df_filtered = df_filtered[df_filtered["day_of_week"].isin(weekday_order)].copy()

# Konversi kategori agar urutannya sesuai
df_filtered["day_of_week"] = pd.Categorical(df_filtered["day_of_week"], categories=weekday_order, ordered=True)

# Pastikan data numerik untuk y-axis
df_filtered[selected_param] = pd.to_numeric(df_filtered[selected_param], errors="coerce")

# Hitung rata-rata untuk setiap hari agar line chart lebih smooth
df_grouped = df_filtered.groupby("day_of_week", observed=True, as_index=False)[selected_param].mean()

chart = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": f"Rata-rata {selected_param} dalam seminggu",
    "mark": {"type": "line", "strokeWidth": 3, "color": "#83c9ff"},
    "encoding": {
        "x": {
            "field": "day_of_week",
            "type": "ordinal",
            "sort": weekday_order,
            "axis": {
                "title": "Hari",
                "labelColor": "#e6eaf1",
                "titleColor": "#e6eaf1",
                "labelAngle": -45
            }
        },
        "y": {
            "field": selected_param,
            "type": "quantitative",
            "axis": {
                "title": selected_param,
                "labelColor": "#e6eaf1",
                "titleColor": "#e6eaf1"
            }
        }
    },
    "data": {"values": df_grouped.to_dict(orient="records")},
    "config": {
        "background": "#0e1117",
        "title": {"color": "#fafafa", "fontSize": 16},
        "axis": {
            "labelFontSize": 12,
            "titleFontSize": 14,
            "gridColor": "#31333F"
        }
    }
}

st.vega_lite_chart(chart, use_container_width=True)

# Tambahkan tabel rata-rata per hari
# st.subheader("Rata-rata per Hari dalam Seminggu")
st.dataframe(df_grouped.style.format({selected_param: "{:.2f}"}))
# Temukan hari dengan nilai tertinggi dan terendah
max_day = df_grouped.loc[df_grouped[selected_param].idxmax(), "day_of_week"]
min_day = df_grouped.loc[df_grouped[selected_param].idxmin(), "day_of_week"]
max_value = df_grouped[selected_param].max()
min_value = df_grouped[selected_param].min()

# Tampilkan kesimpulan otomatis
# st.subheader("Kesimpulan")
st.markdown(f"""
Berdasarkan analisis data, hari dengan **{selected_param} tertinggi** adalah **{max_day}** dengan rata-rata **{max_value:.2f}**.  
Sementara itu, hari dengan **{selected_param} terendah** adalah **{min_day}** dengan rata-rata **{min_value:.2f}**.  
""")
st.markdown("<br><hr style='height:3px;border:none;border-top:3px solid #f0f0f0;'><br>", unsafe_allow_html=True)


# section 4 (comparison)
if location == "Semua":
    st.subheader(f"Perbandingan Kadar {selected_param} antara Changping dan Dongsi")

    if df_filtered.empty:
        st.warning("Data tidak tersedia untuk lokasi ini.")
    else:

        # barplot nilai rata-rata
        max_value = df_filtered[selected_param].max() * 1.1

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_filtered, x="station", y=selected_param, ax=ax, palette="Blues", width=0.5)
        
        ax.set_xlabel("Lokasi")
        ax.set_ylabel(f"Rata-rata {selected_param}")
        ax.set_title(f"Rata-rata {selected_param} Berdasarkan Lokasi")

        st.pyplot(fig)

        # Hitung rata-rata per lokasi
        avg_per_location = df_filtered.groupby("station", observed=True)[selected_param].mean()

        # Tentukan lokasi dengan nilai tertinggi dan terendah
        max_station = avg_per_location.idxmax()
        min_station = avg_per_location.idxmin()
        max_value = avg_per_location.max()
        min_value = avg_per_location.min()

        # Tampilkan kesimpulan
        # st.subheader("Kesimpulan")
        st.markdown(f"""
        Berdasarkan perbandingan kadar **{selected_param}** antara Kota **Changping** dan **Dongsi**,  
        dapat disimpulkan bahwa lokasi dengan **kadar {selected_param} tertinggi** adalah **{max_station}** dengan rata-rata **{max_value:.2f}**.  
        Sementara itu, lokasi dengan **kadar {selected_param} terendah** adalah **{min_station}** dengan rata-rata **{min_value:.2f}**.  

        Perbedaan ini menunjukkan adanya variasi kadar {selected_param} antara kedua lokasi, yang dapat disebabkan oleh faktor lingkungan, aktivitas manusia, atau kondisi cuaca.  
        """)
        # with st.expander("See explanation"):
        #     st.write(
        #         """Visualisasi ini menunjukkan distribusi (boxplot) dan rata-rata (barplot) dari variabel yang dipilih berdasarkan lokasi."""
        #     )

st.markdown("<br><hr style='height:3px;border:none;border-top:3px solid #f0f0f0;'><br>", unsafe_allow_html=True)


# section 5(CLUSTERING)

clustering_features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
df_clustering = df_filtered[clustering_features].dropna()

# Normalisasi data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# menentukan jumlah cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered["Cluster"] = kmeans.fit_predict(df_scaled)

# visualisasi
st.subheader("Hasil Clustering Kualitas Udara")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=df_filtered["PM2.5"], y=df_filtered["PM10"], hue=df_filtered["Cluster"], palette="viridis", alpha=0.6, ax=ax)
plt.xlabel("PM2.5")
plt.ylabel("PM10")
plt.title("Clustering Berdasarkan PM2.5 dan PM10")
st.pyplot(fig)
# Hitung jumlah data di tiap cluster
cluster_counts = df_filtered["Cluster"].value_counts().sort_index()

# Tampilkan kesimpulan
# Hitung jumlah data di tiap cluster
# tampilkan hasil dalam tabel
st.write("Distribusi Cluster:")
st.dataframe(df_filtered[["station", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "Cluster"]].head(20))

cluster_counts = df_filtered["Cluster"].value_counts().sort_index()

# Tampilkan kesimpulan
# st.subheader("Kesimpulan")
st.markdown(f"""
Analisis **K-Means Clustering** membagi data kualitas udara menjadi **3 kelompok utama**, berdasarkan kadar **PM2.5 dan PM10**. Hasilnya adalah:  

- **Cluster 0**: Menunjukkan area dengan **kualitas udara baik** (konsentrasi polutan rendah).  
- **Cluster 1**: Mengindikasikan **kualitas udara sedang**, dengan tingkat polusi yang tidak terlalu tinggi tetapi masih perlu diperhatikan.  
- **Cluster 2**: Merepresentasikan **area dengan polusi tinggi**, yang berpotensi berdampak buruk pada kesehatan manusia.  

**Informasi ini dapat digunakan untuk kebijakan lingkungan, pemantauan polusi, dan peringatan dini bagi masyarakat**
""")

# fig, ax = plt.subplots(figsize=(10,6))
# sns.boxplot(x=df_filtered["Cluster"], y=df_filtered["PM2.5"], ax=ax)
# plt.title("Distribusi PM2.5 di Setiap Cluster")
# st.pyplot(fig)



# st.write("Dashboard Analisis Kualitas Udara")
