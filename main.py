import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
import itertools
from sklearn.cluster import KMeans
import streamlit as st
import io
import warnings
import streamlit.components.v1 as components
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Проверка МНО для ЯК', page_icon='eco.png',layout='wide')
st.title('Генерация Таблицы и Маршрута для Яндекс-Карт по проверкам')

# Инициализация session state для хранения результатов
if 'route_result' not in st.session_state:
    st.session_state.route_result = None
if 'map_html' not in st.session_state:
    st.session_state.map_html = None
if 'excel_bytes' not in st.session_state:
    st.session_state.excel_bytes = None
if 'start_coords' not in st.session_state:
    st.session_state.start_coords = None
if 'calculation_started' not in st.session_state:
    st.session_state.calculation_started = False
if 'final_union_df' not in st.session_state:
    st.session_state.final_union_df = None
if 'final_coords_list' not in st.session_state:
    st.session_state.final_coords_list = None
if 'final_total_distance' not in st.session_state:
    st.session_state.final_total_distance = None

def add_log(message):
    """Добавляет сообщение в лог (внутренний, не отображается)"""
    pass  # Логи больше не нужны для визуализации

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
    lo_proverka['Подпись'] = lo_proverka['Подпись'].astype(str)

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
    merge['Номер метки'] = np.nan
    merge = merge[merge['Подпись'].notnull()]
    merge['Подпись'] = merge['Подпись'].astype(int)
    merge['Широта'] = merge['Широта'].astype(float)
    merge['Долгота'] = merge['Долгота'].astype(float)
    df = merge[['Широта', 'Долгота', 'Описание', 'Подпись', 'Номер метки']]

    #print(lo_proverka)
    #print(graph_df)
    #print(merge)

    # Функция для вычисления расстояния между двумя точками (прямая)
    def distance(point1, point2):
        return geodesic((point1["Широта"], point1["Долгота"]), (point2["Широта"], point2["Долгота"])).meters

    # Функция для получения пешеходного маршрута и расстояния через OSRM (с кэшированием)
    _osrm_cache = {}
    
    def get_osrm_distance(point1, point2, timeout=3):
        """
        Получает расстояние и геометрию пешеходного маршрута между двумя точками через OSRM API.
        Возвращает (расстояние в метрах, список координат маршрута) или (None, None) при ошибке.
        Использует кэширование для ускорения повторных запросов.
        """
        lon1, lat1 = point1["Долгота"], point1["Широта"]
        lon2, lat2 = point2["Долгота"], point2["Широта"]
        
        # Создаем ключ для кэша
        cache_key = (round(lon1, 5), round(lat1, 5), round(lon2, 5), round(lat2, 5))
        if cache_key in _osrm_cache:
            return _osrm_cache[cache_key]
        
        url = f"http://router.project-osrm.org/route/v1/foot/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and len(data['routes']) > 0:
                route = data['routes'][0]
                distance_m = route['distance']
                geometry = route['geometry']['coordinates']  # [[lon, lat], ...]
                # Преобразуем в [lat, lon] для folium
                route_coords = [[coord[1], coord[0]] for coord in geometry]
                result = (distance_m, route_coords)
                _osrm_cache[cache_key] = result
                return result
            else:
                return None, None
        except Exception as e:
            print(f"Ошибка OSRM: {e}")
            return None, None

    # Функция для построения матрицы расстояний с использованием OSRM (оптимизированная версия)
    def build_osrm_distance_matrix(points, max_retries=2, use_parallel=True):
        n = len(points)
        matrix = np.zeros((n, n))
        routes = {}  # Хранение маршрутов для визуализации
        
        # Для больших наборов точек используем упрощенную эвристику
        if n > 50:
            # Для очень больших кластеров используем прямое расстояние с коэффициентом 1.4
            # Это значительно быстрее и дает приемлемые результаты для пешеходов
            for i in range(n):
                for j in range(i + 1, n):
                    direct_dist = distance(points[i], points[j])
                    matrix[i][j] = direct_dist * 1.4
                    matrix[j][i] = direct_dist * 1.4
                    routes[(i, j)] = None
                    routes[(j, i)] = None
            return matrix, routes
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_pairs = n * (n - 1) // 2
        pair_count = 0
        
        if use_parallel and n > 3:
            # Параллельное вычисление матрицы расстояний
            def compute_pair(i_j):
                i, j = i_j
                retry = 0
                while retry < max_retries:
                    dist, route_coords = get_osrm_distance(points[i], points[j])
                    
                    if dist is not None:
                        return (i, j, dist, route_coords)
                    else:
                        retry += 1
                        time.sleep(0.1)
                
                # Если OSRM не ответил, используем прямое расстояние * 1.4 как эвристику
                direct_dist = distance(points[i], points[j])
                return (i, j, direct_dist * 1.4, None)
            
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            with ThreadPoolExecutor(max_workers=min(12, len(pairs))) as executor:
                futures = {executor.submit(compute_pair, pair): pair for pair in pairs}
                
                for future in as_completed(futures):
                    i, j, dist, route_coords = future.result()
                    matrix[i][j] = dist
                    matrix[j][i] = dist
                    routes[(i, j)] = route_coords
                    routes[(j, i)] = route_coords[::-1] if route_coords else None
                    
                    pair_count += 1
                    progress_bar.progress(pair_count / total_pairs)
                    status_text.text(f"Матрица: {pair_count}/{total_pairs}")
        else:
            # Последовательное вычисление для небольших наборов
            for i in range(n):
                for j in range(i + 1, n):
                    retry = 0
                    while retry < max_retries:
                        dist, route_coords = get_osrm_distance(points[i], points[j])
                        
                        if dist is not None:
                            matrix[i][j] = dist
                            matrix[j][i] = dist
                            routes[(i, j)] = route_coords
                            routes[(j, i)] = route_coords[::-1]
                            break
                        else:
                            retry += 1
                            time.sleep(0.1)
                    
                    if dist is None:
                        direct_dist = distance(points[i], points[j])
                        matrix[i][j] = direct_dist * 1.4
                        matrix[j][i] = direct_dist * 1.4
                        routes[(i, j)] = None
                        routes[(j, i)] = None
                    
                    pair_count += 1
                    progress_bar.progress(pair_count / total_pairs)
                    status_text.text(f"Матрица: {pair_count}/{total_pairs}")
        
        progress_bar.empty()
        status_text.empty()
        
        return matrix, routes

    # Алгоритм 2-opt для оптимизации маршрута
    def two_opt(points, distance_matrix, max_iterations=100):
        n = len(points)
        if n <= 2:
            return list(range(n))
        
        # Начальный маршрут: жадный алгоритм (ближайший сосед)
        unvisited = set(range(1, n))
        current = 0
        path = [current]
        
        while unvisited:
            next_point = min(unvisited, key=lambda x: distance_matrix[current][x])
            path.append(next_point)
            unvisited.remove(next_point)
            current = next_point
        
        # Оптимизация 2-opt
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Вычисляем длину текущего маршрута
                    old_cost = distance_matrix[path[i-1]][path[i]] + distance_matrix[path[j]][path[(j+1) % n]]
                    # Вычисляем длину после изменения
                    new_cost = distance_matrix[path[i-1]][path[j]] + distance_matrix[path[i]][path[(j+1) % n]]
                    
                    if new_cost < old_cost:
                        # Разворачиваем участок пути между i и j
                        path[i:j+1] = reversed(path[i:j+1])
                        improved = True
                        break
                
                if improved:
                    break
        
        return path

    # Функция для поиска оптимального пути с использованием 2-opt и OSRM
    def find_optimal_path(df):
        points = df.to_dict('records')
        n = len(points)
        
        if n <= 1:
            return points
        
        if n == 2:
            return points
        
        # Строим матрицу расстояний через OSRM
        add_log(f"Построение матрицы расстояний для {n} точек...")
        distance_matrix, routes = build_osrm_distance_matrix(points)
        
        # Применяем 2-opt для оптимизации
        add_log("Оптимизация маршрута алгоритмом 2-opt...")
        optimal_indices = two_opt(points, distance_matrix)
        
        # Формируем оптимальный путь
        optimal_path = [points[i] for i in optimal_indices]
        
        return optimal_path, routes

    # Производим кластеризацию с заданным числом кластеров
    n_clusters=int(len(df)/4) + 1
    kmeans = KMeans(n_clusters, init = 'k-means++')
    df['Кластер'] = kmeans.fit_predict(df[['Широта','Долгота']])
    
    # Выбор начальной точки: через selectbox или клик на карте
    st.write('**:red[2. Выбери первое МНО, с которого начнешь путь]**')
    
    # Опция выбора начальной точки кликом на карте
    col1, col2 = st.columns([1, 1])
    with col1:
        use_map_click = st.checkbox("Выбрать начальную точку на карте", value=False, key='map_click_option')
    
    if use_map_click and not st.session_state.calculation_started:
        # Отображаем карту для выбора начальной точки ТОЛЬКО если расчет еще не начался
        temp_map = folium.Map(location=[df['Широта'].iloc[0], df['Долгота'].iloc[0]], zoom_start=13)
        for index, row in df.iterrows():
            folium.Marker(
                location=[row['Широта'], row['Долгота']],
                tooltip=f"{row['Подпись']}: {row['Описание']}",
                popup=f"ID: {row['Подпись']}"
            ).add_to(temp_map)
        
        with open('temp_map.html', 'w', encoding='utf-8') as f:
            temp_map.save(f)
        
        with open('temp_map.html', 'r', encoding='utf-8') as f:
            temp_map_html = f.read()
        
        st.write("Нажми на маркер нужной точки на карте:")
        components.html(temp_map_html, width=800, height=600)
        
        # Временное решение: показываем таблицу всех точек для выбора
        st.write("Или выбери из списка ниже:")
        point_list = df['Подпись'].sort_values().unique().tolist()
        option = st.selectbox('Выберите начальную точку:', point_list, key='map_select')
    else:
        option = st.selectbox('**:red[Выбери первое МНО из списка]**', df['Подпись'].sort_values().unique(), key='list_select')
        
    begin_point = option #int(input()) # выберем нужную точку для начала пути
    # Координаты начальной точки
    df_koor_first = df.loc[df['Подпись'] == begin_point,["Широта",'Долгота']]
    # найдем первый кластер - кластер где находится первая точка
    first_cluster = df.loc[df['Подпись'] == begin_point,'Кластер'].values[0]

    df.loc[df['Подпись'] == begin_point, 'First_MNO_klaster'] = 1
    df = df.sort_values('First_MNO_klaster', ascending=False)

    # Устанавливаем флаг начала расчета - теперь расчет начался
    st.session_state.calculation_started = True
    add_log(f"Начало расчета маршрута. Точек: {len(df)}, Кластеров: {n_clusters}")

    iter_Cluster = first_cluster
    union_df = pd.DataFrame()

    for_pred_len = 0
    
    # Общий прогресс-бар для всего процесса
    total_clusters = n_clusters
    overall_progress_bar = st.progress(0)
    overall_status = st.empty()

    for i in range(1,n_clusters+1):
        overall_status.text(f"Общий прогресс: {i-1}/{total_clusters} кластеров обработано")
        overall_progress_bar.progress((i-1) / total_clusters)
        
        dt = df.loc[df['Кластер'] == iter_Cluster] # отделяем текущий кластер
        length_dt = len(dt) # кол-во элементов в кластере
        add_log(f"Обработка кластера {i}: {length_dt} точек")
        
        # Находим оптимальный путь
        result = find_optimal_path(dt)
        
        # Обрабатываем результат: если это кортеж (путь, маршруты), берем только путь
        if isinstance(result, tuple):
            optimal_path, cluster_routes = result
        else:
            optimal_path = result
            cluster_routes = {}

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

    # Завершаем общий прогресс
    overall_progress_bar.progress(1.0)
    overall_status.text("Общий прогресс: завершен!")

    del union_df['First_MNO_klaster']
    union_df = union_df.reset_index().drop('index',axis=1)

    # Сохраняем результаты в session state для предотвращения дублирования и повторного расчета
    st.session_state.final_union_df = union_df.copy()
    add_log(f"Результат сохранен. Строк: {len(union_df)}")


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

    # Добавление точек на карту с разными цветами для разных кластеров и номерами меток
    for index, row in union_df.iterrows():
        # Создаем HTML для метки с порядковым номером
        marker_number = int(row['Номер метки'])
        folium.Marker(
            location=[row['Широта'], row['Долгота']],
            popup=f"Номер метки: {marker_number}\n{row['Описание']}",
            tooltip=f"#{marker_number}: {row['Подпись']}",
            icon=folium.DivIcon(
                html=f'''<div style="font-size: 10pt; font-weight: bold; color: white; background-color: {cluster_colors_dict.get(row['Кластер'], 'gray')}; border: 2px solid black; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px;">{marker_number}</div>''',
                icon_size=(24, 24)
            )
        ).add_to(m)

    # Построение пешеходных маршрутов между точками через OSRM
    points_sorted = union_df.sort_values(by='Номер метки')
    coords_list = points_sorted[['Широта', 'Долгота']].values.tolist()
    
    total_osrm_distance = 0
    
    st.write("🗺️ Построение пешеходных маршрутов на карте...")
    route_progress = st.progress(0)
    
    for i in range(len(coords_list) - 1):
        point1 = points_sorted.iloc[i]
        point2 = points_sorted.iloc[i + 1]
        
        # Получаем маршрут через OSRM
        dist, route_coords = get_osrm_distance(point1, point2, timeout=3)
        
        if route_coords is not None and len(route_coords) > 1:
            # Рисуем детальный пешеходный маршрут
            folium.PolyLine(
                locations=route_coords,
                color='red',
                weight=4,
                opacity=0.8
            ).add_to(m)
            total_osrm_distance += dist / 1000  # в км
        else:
            # Если OSRM не ответил, рисуем прямую линию
            folium.PolyLine(
                locations=[coords_list[i], coords_list[i + 1]],
                color='red',
                weight=3,
                opacity=0.6,
                dash_array='5'
            ).add_to(m)
            direct_dist = geodesic(coords_list[i], coords_list[i + 1]).km
            total_osrm_distance += direct_dist
        
        route_progress.progress((i + 1) / (len(coords_list) - 1))
    
    route_progress.empty()
    
    total_distance = round(total_osrm_distance, 2)

    print(f'Общий путь: {total_distance} км')
    # Вывод на экран общего пути
    st.write(f'**Количество МНО: {union_df["Долгота"].count()} шт.. Общий путь: {total_distance} км**')
    
    # Сохраняем результаты в session state чтобы избежать повторного расчета при скачивании
    if 'route_result' not in st.session_state or st.session_state.route_result is None:
        st.session_state.route_result = {
            'union_df': union_df,
            'total_distance': total_distance
        }

    m.save('map.html')

    with open('map.html', 'r', encoding='utf-8') as f:
        map_html = f.read()

    itog_table = union_df.drop(columns ='Кластер')
    
    towrite = io.BytesIO()
    itog_table.to_excel(towrite, index=False)
    towrite.seek(0)  # возврат указателя в начало байтового объекта
    excel_bytes = towrite.read()
    
    # Сохраняем в session state для использования кнопкой скачивания
    st.session_state.map_html = map_html
    st.session_state.excel_bytes = excel_bytes
    st.session_state.itog_table = itog_table
    st.session_state.final_coords_list = coords_list  # Сохраняем координаты для экспорта в навигатор
    st.session_state.final_total_distance = total_distance
    
    add_log(f"Маршрут построен. Длина: {total_distance} км, Точек: {len(coords_list)}")

    # Отображение карты
    components.html(st.session_state.map_html, width=1700, height=800, scrolling=False)

    # Отображение таблицы
    st.write(st.session_state.itog_table)
    
    # Экспорт в навигатор
    st.subheader("🧭 Экспорт в навигатор")
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        # Формируем GPX файл для навигатора
        gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        gpx_content += '<gpx version="1.1" creator="RouteGenerator">\n'
        gpx_content += '<trk>\n<trkseg>\n'
        
        for coord in coords_list:
            lat, lon = coord
            gpx_content += f'<trkpt lat="{lat}" lon="{lon}"></trkpt>\n'
        
        gpx_content += '</trkseg>\n</trk>\n</gpx>'
        
        gpx_bytes = gpx_content.encode('utf-8')
        
        st.download_button(
            label='📥 Скачать GPX (для навигатора)',
            data=gpx_bytes,
            mime='application/gpx+xml',
            file_name='route.gpx',
            key='download_gpx'
        )
        st.info("GPX файл можно открыть в Яндекс.Навигаторе, Google Maps, OsmAnd или другом навигаторе")
    
    with col_nav2:
        # Дополнительно: KML для Google Earth
        kml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        kml_content += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        kml_content += '<Document>\n<Placemark>\n<LineString>\n<coordinates>\n'
        
        for coord in coords_list:
            lat, lon = coord
            kml_content += f'{lon},{lat},0\n'
        
        kml_content += '</coordinates>\n</LineString>\n</Placemark>\n</Document>\n</kml>'
        
        kml_bytes = kml_content.encode('utf-8')
        
        st.download_button(
            label='🌍 Скачать KML (Google Earth)',
            data=kml_bytes,
            mime='application/vnd.google-earth.kml+xml',
            file_name='route.kml',
            key='download_kml'
        )

    # Кнопка скачивания использует сохраненные данные из session state
    st.download_button(label='**:red[Скачать xlsx]**'\
                       , data=st.session_state.excel_bytes
                       ,mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                       ,file_name='Для ЯК.xlsx'
                       ,key='download_xlsx'
                       )
    st.markdown("[ЯКонструктор](https://yandex.ru/map-constructor)")



# streamlit run main.py