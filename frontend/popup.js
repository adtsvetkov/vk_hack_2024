document.addEventListener('DOMContentLoaded', async () => {
  // Кнопка "Закрыть"
  document.getElementById('close-btn').addEventListener('click', () => {
    window.close(); // Закрывает popup
  });

  // Кнопка "Обновить"
  document.getElementById('refresh-btn').addEventListener('click', async () => {
    await updateAggressionLevel(); // Перезапрашивает данные и обновляет интерфейс
  });

  // Инициализация при загрузке
  await updateAggressionLevel();
});

// Функция обновления уровня агрессии
async function updateAggressionLevel() {
  const aggressionLevel = await getAggressionLevel(); // Получение уровня агрессии (замените на реальную функцию)

  // Обновление текста заголовка
  const header = document.getElementById('header');
  header.textContent = `Уровень агрессии: ${aggressionLevel}`;

  // Обновление картинки в зависимости от уровня
  const indicator = document.getElementById('indicator');
  if (aggressionLevel > 85) {
    indicator.src = 'img/Dokhlik.png'; // Смерть уровень
  } else if (aggressionLevel > 75) {
    indicator.src = 'img/Plokho.png'; // Высокий уровень
  } else if (aggressionLevel > 40) {
    indicator.src = 'img/Pofig.png'; // Средний уровень
  } else if (aggressionLevel > 20) {
    indicator.src = 'img/Dovolen.png'; // Низкий уровень
  } else {
    indicator.src = 'img/Schastliv.png'; // Очень низкий уровень
  }
}

// Мок-функция для получения данных (замените на реальную интеграцию)
async function getAggressionLevel() {
  return Math.floor(Math.random() * 100); // Симуляция уровня
}

let mediaRecorder;
let isRecording = false;

document.addEventListener('DOMContentLoaded', () => {
  // Получаем кнопки
  const startButton = document.getElementById('start-recording');
  const stopButton = document.getElementById('stop-recording');

  // Проверка, найдены ли кнопки
  if (!startButton || !stopButton) {
    console.error('Элементы start-recording или stop-recording не найдены в DOM!');
    return;
  }

  // Обработчик кнопки "Начать запись"
  startButton.addEventListener('click', () => {
    if (!isRecording) {
      isRecording = true;
      startButton.disabled = true;
      stopButton.disabled = false;

      // Получаем текущую вкладку
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const currentTab = tabs[0];

        // Захватываем звук текущей вкладки
        chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
          if (chrome.runtime.lastError || !stream) {
            console.error('Ошибка захвата звука:', chrome.runtime.lastError?.message || 'Нет потока');
            isRecording = false;
            startButton.disabled = false;
            stopButton.disabled = true;
            return;
          }

          // Настроим MediaRecorder для записи
          mediaRecorder = new MediaRecorder(stream);
          let audioChunks = [];

          // Когда есть данные (части записи)
          mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
              audioChunks.push(event.data);

              // Если прошло 5 секунд, сохраняем аудио
              if (isRecording) {
                saveAudioChunk(audioChunks);
                audioChunks = [];
              }
            }
          };

          // Начинаем запись с интервалом 5 секунд
          mediaRecorder.start(5000);
          console.log('Запись началась.');
        });
      });
    }
  });

  // Обработчик кнопки "Остановить запись"
  stopButton.addEventListener('click', () => {
    if (isRecording) {
      isRecording = false;
      startButton.disabled = false;
      stopButton.disabled = true;

      // Останавливаем запись
      mediaRecorder.stop();
      console.log('Запись остановлена.');
    }
  });

  // Функция для сохранения аудиочанка
  function saveAudioChunk(audioChunks) {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const fileName = `audio_${Date.now()}.wav`;

    localStorage.setItem(fileName, JSON.stringify(audioChunks));

    // Отправляем файл на сервер для анализа
    sendToPython(audioBlob, fileName);
}

  // Пример функции для передачи пути к файлу в Python
  // Пример функции для передачи пути к файлу в Python
function sendToPython(fileBlob, fileName) {
  const formData = new FormData();
  formData.append('file', fileBlob, fileName);

  fetch('http://localhost:5000/analyze', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      console.log('Результат анализа:', data);
      // Здесь можно обновить интерфейс на основе полученных данных
  })
  .catch(error => {
      console.error('Ошибка при отправке файла:', error);
  });

  alert(data)

  const header = document.getElementById('header');
  header.textContent = `Уровень агрессии: ${data}`;
}
});