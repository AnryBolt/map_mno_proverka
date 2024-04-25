import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
import itertools
from sklearn.cluster import KMeans
import streamlit as st
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Проверка МНО для ЯК', page_icon='eco.png',layout='wide')
st.title('Генерация Таблицы и Маршрута для Яндекс-Карт по проверкам')

uploaded_lo_proverka = st.file_uploader("**:red[1. Выбери файл ЛО_Проверка...]**")

if uploaded_lo_proverka != None:
    lo_proverka = pd.read_excel(uploaded_lo_proverka,usecols=['2. ID МНО','5. МНО / Широта', '6. МНО / Долгота', '8. Балансодержатель МНО', '9. Категория МНО']).drop(0)
    lo_proverka.columns=['Подпись', 'Широта', 'Долгота', 'Описание', 'Категория']
    lo_proverka['Категория'] = lo_proverka['Категория']\
        .str.replace('Бункерная площадка','БП')\
        .str.replace('Контейнерная площадка','КП')
    lo_proverka['ВМР'] = 'ВМР'
    lo_proverka['ТКО'] = 'ТКО'
    lo_proverka = lo_proverka.drop_duplicates('Подпись')

    graph_df = pd.read_excel('Graphic.xlsm', usecols=['Идентификатор ТСОО', 'Прибытие по', 'ВМР'])
    graph_df.columns=['МНО', 'Прибытие_по', 'Категория_Конт']
    graph_df = graph_df.drop_duplicates()
    graph_df['МНО'] = graph_df.loc[:,'МНО'].astype(str)
    graph_df['Прибытие_по'] = graph_df['Прибытие_по'].astype(str).str[:-3]

    merge = lo_proverka.merge(graph_df,left_on=['Подпись','ВМР'], right_on=['МНО','Категория_Конт'],how='left')\
        .merge(graph_df,left_on=['Подпись','ТКО'], right_on=['МНО','Категория_Конт'],how='left')\
        .drop(['ВМР','ТКО','МНО_x','МНО_y','Категория_Конт_y','Категория_Конт_x'], axis=1)

    merge['Описание'] = merge['Категория'] + "; " + "ВМР: " + merge['Прибытие_по_x'].astype('str') + "; " + "ТКО: " + merge['Прибытие_по_y'].astype('str') + "; " + merge['Описание']
    merge = merge.drop(["Категория","Прибытие_по_x","Прибытие_по_y"], axis=1)
    merge['Номер метки'] = np.NAN
    merge = merge[merge['Подпись'].notnull()]
    merge['Подпись'] = merge['Подпись'].astype(int)
    merge['Широта'] = merge['Широта'].astype(float)
    merge['Долгота'] = merge['Долгота'].astype(float)
    df = merge[['Широта', 'Долгота', 'Описание', 'Подпись', 'Номер метки']]

    # Функция для вычисления расстояния между двумя точками
    def distance(point1, point2):
        return geodesic((point1["Широта"], point1["Долгота"]), (point2["Широта"], point2["Долгота"])).meters

    # Функция для поиска оптимального пути
    def find_optimal_path(df):
        points = df.to_dict('records')
        optimal_path = []
        min_distance = float('inf')
        for path in itertools.permutations(points[1:], len(points) - 1):
            current_path = [points[0]] + list(path)
            total_distance = sum(distance(current_path[i], current_path[i+1]) for i in range(len(current_path) - 1))
            if total_distance < min_distance:
                min_distance = total_distance
                optimal_path = current_path
        return optimal_path

    # Производим кластеризацию с заданным числом кластеров
    n_clusters=int(len(df)/4)
    kmeans = KMeans(n_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    df['Кластер'] = kmeans.fit_predict(df[['Широта','Долгота']])
    m = folium.Map(location=[df['Широта'].iloc[0], df['Долгота'].iloc[0]], zoom_start=13) # отобразим все точки
    for index, row in df.iterrows():
        folium.Marker(
            location=[row['Широта'], row['Долгота']],
            tooltip=row['Подпись'],
        ).add_to(m)

    # Выпадающий список МНО для выбора первого МНО
    option = st.selectbox('**:red[2. Выбери первое МНО, с которого начнешь путь]**',df['Подпись'].sort_values().unique())
        
    begin_point = option #int(input()) # выберем нужную точку для начала пути
    # Координаты начальной точки
    df_koor_first = df.loc[df['Подпись'] == begin_point,["Широта",'Долгота']]
    # найдем первый кластер - кластер где находится первая точка
    first_cluster = df.loc[df['Подпись'] == begin_point,'Кластер'].values[0]

    df.loc[df['Подпись'] == begin_point, 'First_MNO_klaster'] = 1
    df = df.sort_values('First_MNO_klaster', ascending=False)

    iter_Cluster = first_cluster
    union_df = pd.DataFrame()

    for_pred_len = 0

    for i in range(1,n_clusters+1):
        dt = df.loc[df['Кластер'] == iter_Cluster] # отделяем текущий кластер
        length_dt = len(dt) # кол-во элементов в кластере
        # Находим оптимальный путь
        optimal_path = find_optimal_path(dt)

        # Обновляем DataFrame с найденным оптимальным путем
        dt_optimal = pd.DataFrame(optimal_path)

        begin = len(union_df) + 1 # первый итератор
        end = begin + len(dt_optimal) # конечный итератор
        dt_optimal["Номер метки"] = range(begin,end)

        for_pred_len = len(dt_optimal)

        union_df = pd.concat([union_df, dt_optimal]) # поместим датафрейм текущего кластера в общий новый датафрейм
        df_koor_last = dt_optimal[["Широта",'Долгота']].iloc[[-1]] # координаты последней точки в датафрейме

        df = df.loc[df['Кластер'] != iter_Cluster] # обрезаем из главного датафрейма текущий кластер
        if len(df) == 0:
            break
        df['Расстояние_от_пред_кластера'] = df.apply(distance,args=(df_koor_last.iloc[0],),axis=1)

        df_near_point = df.loc[df['Расстояние_от_пред_кластера'] == df['Расстояние_от_пред_кластера'].min()].head(1) # точка с наименьшим расстоянием
        iter_Cluster = df_near_point['Кластер'].iloc[0] # следующий обрабатываемый кластер
        first_mno_klaster = df_near_point['Подпись'].iloc[0]
        df['First_MNO_klaster'] = 0
        del df['Расстояние_от_пред_кластера']
        df.loc[df['Подпись'] == first_mno_klaster, 'First_MNO_klaster'] = 1
        df = df.sort_values('First_MNO_klaster', ascending=False)

    del union_df['First_MNO_klaster']
    union_df = union_df.reset_index().drop('index',axis=1)


    # Создание карты
    m = folium.Map(location=[union_df['Широта'].iloc[0], union_df['Долгота'].iloc[0]], zoom_start=13)

    # Создание словаря цветов для разных кластеров
    cluster_colors = [
        'red',       # Красный
        'green',     # Зеленый
        'blue',      # Синий
        'yellow',    # Желтый
        'purple',    # Фиолетовый
        'orange',    # Оранжевый
        'pink',      # Розовый
        'cyan',      # Голубой
        'brown',     # Коричневый
        'black',     # Черный
        'white',     # Белый
        'tomato',    # Томатный
        'purple',    # Фиолетово-синий
        'teal',      # Темно-зеленый
        'orangered', # Огненно-красный
        'olive',     # Оливковый
        'steelblue', # Стальной синий
        'goldenrod', # Золотой
        'seagreen',  # Морской зеленый
        'hotpink'    # Горчичный
    ]

    cluster_colors_dict = dict(zip(range(len(cluster_colors)), cluster_colors))

    # Добавление точек на карту с разными цветами для разных кластеров
    for index, row in union_df.iterrows():
        folium.Marker(
            location=[row['Широта'], row['Долгота']],
            popup=f"Номер метки: {row['Номер метки']}\n{row['Описание']}",
            tooltip=row['Подпись'],
            icon=folium.Icon(color=cluster_colors_dict.get(row['Кластер'], 'gray'))
        ).add_to(m)

    # Создание красных линий соединения точек по номеру метки
    points = union_df.sort_values(by='Номер метки')

    folium.PolyLine(
        locations=points[['Широта', 'Долгота']],
        color='red'
    ).add_to(m)  # Добавление линии на карту

    coords = points[['Широта', 'Долгота']].values.tolist()

    # Вычислите общее расстояние, суммируя расстояния между всеми последовательными точками
    total_distance = round(sum(geodesic(coords[i], coords[i+1]).km for i in range(len(coords)-1)),2)

    print(f'Общий путь: {total_distance} км')
    # Вывод на экран общего пути
    st.write(f'**Общий путь: {total_distance} км**')

    m.save('map.html')

    with open('map.html', 'r', encoding='utf-8') as f:
        map_html = f.read()

    # Отображение карты
    st.components.v1.html(map_html, width=1700, height=800)

    itog_table = union_df.drop(columns ='Кластер')
    # Отображение таблицы
    st.write(itog_table,)

    towrite = io.BytesIO()
    itog_table.to_excel(towrite, index=False)
    towrite.seek(0)  # возврат указателя в начало байтового объекта
    excel_bytes = towrite.read()

    st.download_button(label='**:red[Скачать xlsx]**'\
                       , data=excel_bytes
                       ,mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                       ,file_name='Для ЯК.xlsx'
                       )

    st.markdown("[ЯКонструктор](https://yandex.ru/map-constructor)")

# streamlit run main.py