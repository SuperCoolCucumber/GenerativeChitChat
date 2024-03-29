# Generative Chit Chat

### Модель и данные
Модель основана на архитектуре `transformer` и инициализирована с помощью библиотеки [x-transformers](https://github.com/lucidrains/x-transformers).
Данные для обучения: вопросы-ответы из сервиса Ответы@Mail.ru -- 300к строк (неполный датасет).

### Железо
Было много попыток обучения на коротких данных (~1000 строк) для файн-тьюнинга модели. \
В целом вышло около 20 часов обучения на _nVidia GeForce RTX 3090_.


### Проблемы во время обучения
- loss выходил на плато, стабильно и быстро снижаясь с 12 до 6 и оставался на 6, никак не снижаясь дальше (**решение**: подкручивание learning rate [не помогло] и смена архитктуры с `decoder-only` на `encoder-decoder`)
- модель стала генерировать набор символов из разных языков одновременно (**решение**: применение causal masking с помощью класса `AutoregressiveWrapper`)
- ответы модели стали похожи на реальный язык: она иногда стала генерировать нормальные человеческие слова, однако все остальные слова в ее ответе не имеют смысла и похожи на набор букв (**решение**: набор букв, скорее всего, -- это следствие `BertTokenizer`, который делит текст не на слова, а на кусочки слов чаще всего. Поэтому было принято решение обучать модель в течение примерно 10 часов на больших данных.)


### Примеры ответов модели
И история того, как она обучалась и становилась (или не становилась) умнее лежат в файле `v2.log`.

В начале обучения заметно, как модель обучилась скобкам -- самому популярному способу выражению эмоций в рунете )))))))).\
Примерно к 30-й эпохе она стала пытаться применять хэштэги. \
Loss медленно, но стабильно снижался с 6 до 5.11 (пока что). \
К 60-й эпохе стали все чаще появляться реальные человеческие слова в ответе.

### Планы на будущее
Похоже, идея с использованием сторонней библиотки для обучения трансформера не дала таких хороших результатов, как хотелось бы.

Поэтому стоит заняться созданием и обучением собственной модели на pytorch. 

### [Телеграм бот](https://t.me/gobbledygook_bot) (приостановлен)

Проект выполнен в рамках третьего домашнего задания курса Deep Learning in NLP на магистерской программе "Компьютерная лингвистика" ВШЭ.

