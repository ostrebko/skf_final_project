  Работа по написанию кода для парсинга характеристик домов была сделана в Google Colab (при желании заменой нескольких строк код перенести на локальный ноутбук или в другую облачную среду). Парсинг данных проводился с сайта https://mingkh.ru/ ("МинЖКХ - некоммерческий общественный инициативный проект повышения общественной осведомлённости в области функционирования управляющих компаний и ТСЖ"). Прим.: На сайте представлены города России. 
  
  Для предварительного анализа первоначально были выбраны два города, в каждом из которых проживает более 1 млн человек: г.Нижний Новгород (*city_1*) и г.Казань (*city_2*).  
  Парсинг данных проводился в два этапа: 1 этап - парсинг ссылок на все дома рассматриваемого города (файлы: *houses_links_full_1.txt*, *houses_links_full_2.txt*); 2 этап - проход по всем ссылкам с непосредственным парсингом данных по домам. Добавлено отображение общего числа обработанных ссылок (обновление через каждые 100 ссылок). В процессе парсинга характеристик домов  сервер обрывал соединение. Для продолжения парсинга были добавлены и закомментированы несколько строк, позволяющих отступить в файле со ссылками нужное количество строк и продолжить парсинг (В предваритеном анализе получение нужного количества строк для пропуска не было автоматизировано и выяснялось открытием файла, в который добавлялись данные. При необходимости можно добавить соответствующую функцию). Данные при парсинге собирались в список, после чего были сохранены в файл в формате csv (*houses_1.csv*, *houses_2.csv*).  Собранные данные сохранены в *Houses/city_1*, *Houses/city_2*.
  
  Далее в соответствии с выбранной задачей был проведен предварительный анализ данных по сериям домов. С помощью библиотеки Pandas были подсчитаны количество уникальных значений по сериям домов (метод value_counts). Для города Нижний Новгород в столбце *series_of_house* ('Серия, тип постройки') число данных по сериям домов, которым соотвествовали значения: 'отсутствует', 'кирпичный', '-', 'нет', 'нет данных', 'нет данных.', 'панельный', 'дореволюционный', 'индивидуальный', 'народная стройка', 'блочный', 'деревянный времён первых пятилеток', 'индивид' превышает более 80% всех данных. По городу Казань: ситуация лучше, но число данных по сериям домов, которым соотвествовали значения: 'индивид.', 'отсутствует', 'неопределен' превышает 38%. Что является довольно большим значением. 
  
  В связи с большим отличием по доле нерелевантных значений, был дополнительно проанализирован город Новосибирск (крупный город с числом жителей более 1 млн. чел. в регионе, удаленном от рассматриваемых выше городов). Число данных по сериям домов, которым соотвествовали значения: 'отсутствует' (или аналогичные данные) превышает 63%. Собранные данные сохранены в *Houses/city_3*
  
  Все полученные данные сохранены в архиве *Houses.rar*
  
  Результаты анализа данных показали, что в рассмотренных данных по выбранной задаче (предсказание серий домов) был получен плохой таргет для обучения модели. Выбор иного таргета не предполагался, так как основной интерес для компании представляют сведения по сериям домов (планировкам квартир). На основании проведенного анализа было решено отказаться от решения первой возможной задачи. Однако, важно отметить, что полученные данные могут быть использованы как дополнительные данные при обучении иных моделей, так как содержат большое число дополнительных данных по домам в городах. Собранные данные можно учесть в других задачах, связанных, например, с оценкой стоимости или аренды жилья.