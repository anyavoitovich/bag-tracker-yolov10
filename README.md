### Bag Detection and Counting on the Conveyor Belt

#### Task Description
The task involves detecting and counting bags on a conveyor belt that can move in both directions. It is necessary to ensure an accurate count of bags only in one direction. The YOLOv10s model, trained for bag detection, was used for this purpose. The input test video has dimensions of 640x340 pixels.

#### Architecture and Technology
- **Model**: YOLOv10s trained for bag detection.
- **Framework**: OpenCV for image and video processing, YOLO for object detection, and `filterpy` library for Kalman filters.
- **Docker**: All stages of the development and task execution were assembled using Docker, providing a stable execution environment and ease of deployment.

#### Task Execution Scheme:
1. **Training the YOLOv10s Model**:
   - The model was trained using PyTorch and YOLO libraries, with weights saved in the folder `train5`.
   - The model weights, trained using this data, are stored in `train5/weights/best.pt`.

2. **Setting up OpenCV and Kalman Filter**:
   - Video file opened using `cv2.VideoCapture`.
   - Video parameters set (width, height, FPS).
   - A line is defined on the screen to count bags (horizontal line at 250 pixels vertically).
   - Kalman filters initialized for tracking objects on the video.

3. **Object Detection and Filtering**:
   - Use YOLO model for detecting bags on each frame.
   - Filter objects by minimum size (width and height more than 50 pixels).
   - Apply the Intersection over Union (IoU) method to eliminate duplicate detections.
   - A new tracker is created for each object, and a Kalman filter is initialized to track the object on the frame.

4. **Line Crossing Check**:
   - For tracked objects, check for crossing the line at 250 pixels.
   - If an object crosses the line from top to bottom, the `in` (entry) counter increases. If from bottom to top — the `out` (exit) counter increases.

5. **Handling Lost Trackers**:
   - If an object is lost (unable to track), the Kalman filter is used to predict its position.
   - If an object cannot be tracked for more than `max_lost_frames`, it is removed from the tracker.

6. **Results and Output**:
   - The `in` and `out` counters are updated on each frame.
   - The final value (difference `out` - `in`) is displayed on each frame.
   - The results are shown on the video as a green text field.

7. **Documentation and Docker Packaging**:
   - The project was packaged using Docker, which allowed for a stable environment to perform object detection and tracking tasks.
   - All necessary dependencies, including OpenCV, YOLO, and `filterpy`, were installed in the Docker image.

### Обнаружение и подсчёт мешков на конвейере

#### Описание задачи
Задача состоит в обнаружении и подсчёте мешков на конвейере, который может двигаться в обе стороны. Необходимо обеспечить точный подсчёт мешков только в одном направлении. Для этого использовалась модель YOLOv10s, обученная на детекцию мешков. Входное тестовое видео имеет размеры 640x340 пикселей.

#### Архитектура и технология
- **Модель**: YOLOv10s, обученная для детекции мешков.
- **Фреймворк**: OpenCV для работы с изображениями и видео, YOLO для детекции объектов, библиотека `filterpy` для Калмановских фильтров.
- **Докер**: Все этапы разработки и выполнения задачи были собраны с использованием Docker, обеспечивая стабильную среду выполнения и удобство развёртывания.

#### Схема выполнения задачи:
1. **Обучение модели YOLOv10s**:
   - Модель была обучена на базе библиотек PyTorch и YOLO, используя веса, сохранённые в папке `train5`.
   - Веса модели, обученные с использованием этих данных, хранятся в `train5/weights/best.pt`.

2. **Настройка OpenCV и Kalman Filter**:
   - Открытие видеофайла с использованием `cv2.VideoCapture`.
   - Установка параметров видео (ширина, высота, FPS).
   - Определение линии на экране для подсчёта мешков (горизонтальная линия на 250 пикселей по вертикали).
   - Инициализация Kalman фильтров для отслеживания объектов на видео.

3. **Обнаружение и фильтрация объектов**:
   - Использование модели YOLO для обнаружения мешков на каждом кадре.
   - Фильтрация объектов по минимальному размеру (ширина и высота более 50 пикселей).
   - Применение метода пересечения IoU (Intersection over Union) для устранения дублирующихся обнаружений.
   - Для каждого объекта создаётся новый трекер, инициализируется Kalman фильтр, и объект отслеживается на кадре.

4. **Проверка пересечения линии**:
   - Для отслеживаемых объектов проверяется пересечение с линией на 250 пикселей.
   - Если объект пересекает линию сверху вниз, увеличивается счётчик `in` (вход). Если снизу вверх — счётчик `out` (выход).

5. **Обработка потерь трекеров**:
   - Если объект теряется (либо не удаётся его отследить), используется Kalman фильтр для прогнозирования его положения.
   - Если объект не удаётся отследить более `max_lost_frames`, он удаляется из трекера.

6. **Результаты и вывод**:
   - Счётчики `in` и `out` обновляются на каждом кадре.
   - Выводим итоговое значение (разница `out` - `in`) на каждом кадре.
   - Результаты отображаются на видео в виде зелёного текстового поля.

7. **Документирование и сборка с использованием Docker**:
   - Проект был собран с использованием Docker, что позволило создать стабильную среду для выполнения задач детекции и отслеживания объектов.
   - Все необходимые зависимости, включая OpenCV, YOLO, и `filterpy`, были установлены в Docker образе.
