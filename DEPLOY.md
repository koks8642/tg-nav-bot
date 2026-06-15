# Деплой RQM-бота на бесплатный хост

Бот — это **поллинг** (постоянно опрашивает Telegram), значит ему нужен хост,
который работает **24/7** и **не блокирует** `api.telegram.org`, `telegra.ph`,
`teletype.in`. Бесплатные «спящие» платформы (Render free, Replit) и площадки с
белым списком исходящих (PythonAnywhere free) **не подходят**.

## Лучший вариант: Oracle Cloud — Always Free

**Почему он:** по-настоящему бесплатно навсегда (не триал), полноценная VM с
рут-доступом, свой диск под SQLite, дата-центры за пределами РФ (полный доступ к
Telegram/Telegraph/Teletype), Docker, мощности с запасом под сборку скачиваний.

- Шейп **Ampere ARM (VM.Standard.A1.Flex)** — до 4 ядра / 24 ГБ RAM бесплатно
  (бери 1 ядро / 6 ГБ — за глаза). Если ARM-ёмкости в регионе нет, возьми
  **VM.Standard.E2.1.Micro** (AMD, 1 ГБ) — боту хватит.
- Регион: **Frankfurt / Amsterdam / любой не-РФ**.
- При регистрации нужна карта (только для верификации, на Always Free не спишут).

**Запасной вариант:** любой небольшой VPS с Docker и постоянным диском под SQLite.

---

## Шаги (Oracle, Ubuntu 22.04, Docker)

### 1. Создать VM
1. Зарегистрируйся на cloud.oracle.com, выбери **Always Free** регион вне РФ.
2. Compute → Instances → **Create instance**.
   - Image: **Ubuntu 22.04**.
   - Shape: **Ampere A1 Flex**, 1 OCPU / 6 GB (или E2.1.Micro, если A1 занят).
   - Сохрани приватный SSH-ключ.
3. После создания запиши **публичный IP**.

### 2. Зайти и поставить Docker
```bash
ssh -i путь/к/ключу ubuntu@ПУБЛИЧНЫЙ_IP
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER && newgrp docker     # docker без sudo
```

Для GitHub Actions лучше сделать отдельного deploy-пользователя:
```bash
sudo adduser --disabled-password --gecos "" deploy
sudo usermod -aG docker deploy
sudo mkdir -p /home/deploy/.ssh
sudo nano /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh
sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
```

### 3. Забрать код
```bash
git clone https://github.com/koks8642/tg-nav-bot.git rqm
cd rqm
```
> Образ собирается «чистым» сам: в контейнер попадает только `app/` + зависимости.
> Тесты и operator-скрипты в контейнер **не идут** — ничего вручную чистить не надо.

### 4. Заполнить секреты `.env`
```bash
cp .env.example .env
nano .env
chmod 600 .env        # важно: только владелец читает токен
```
Заполни:
- `BOT_TOKEN=` — токен у @BotFather (бот должен быть **админом канала**).
- `CHANNEL_CHAT_ID=` — id канала (формат `-100…`). Если не знаешь — запусти
  локально `python -m scripts.whoami` и напиши в канал, либо спроси у @userinfobot.
- `OWNER_USER_IDS=` — твой Telegram user_id (можно несколько через запятую).
- `TELEGRAPH_TOKEN=` — необязательно: если пусто, бот создаст аккаунт сам при
  первом старте и сохранит токен в БД. В логи токен не выводится; если хочешь
  хранить его именно в `.env`, заранее запусти `python -m scripts.setup_telegraph`.
- `SEED_DEFAULT_REGISTRY=0` — для чистого старта без встроенных RQM-проектов,
  разделов и хэштегов. Для старого RQM-режима оставь `1`.

### 5. Запустить
```bash
docker compose up -d --build
docker compose logs -f           # смотрим старт
```
Должны увидеть: `Bot polling started`, `command menu set`, `download worker
started`. **Токен в логах не светится** — это нормально, так и задумано.

### 6. После старта
1. Если `SEED_DEFAULT_REGISTRY=1`, БД один раз засеет дефолтный реестр RQM
   (проекты + хэштеги). Если `SEED_DEFAULT_REGISTRY=0`, бот стартует полностью
   чистым, и структуру нужно создать через Админку.
2. В боте: **🛠 Админка → 🔗 Ссылки** → возьми ссылку на **главную навигацию** →
   **закрепи её в канале**.
3. Дальше наполняй вживую (см. ниже).

---

## Твой рабочий процесс (пустая БД → наполнение вживую)

При запущенном боте просто **постишь/редактируешь посты в канале** с хэштегами —
бот ловит их и сам строит навигацию (в течение ~3 секунд):

- `#вид #тайтл` + ссылка «Глава N» (Telegraph для новелл, Teletype для манги) →
  глава встаёт в проект. Пример: `#новелла #покровитель` + ссылка `Глава 305`.
- `#арты` / `#мемы` / `#заметки` → пост попадает в раздел.
- **Виды** (Манга/Манхва/Новеллы) и привязку хэштегов к проектам делаешь один
  раз в **Админке**, если их нет в засеянном реестре.

> Для полностью пустой БД без дефолтных проектов RQM держи
> `SEED_DEFAULT_REGISTRY=0` в `.env`.

---

## Эксплуатация

| Задача | Как |
|---|---|
| Логи | `docker compose logs -f` (токен скрыт) |
| Бэкап БД | в боте: **Админка → 💾 Скачать бэкап БД** (придёт файлом в личку). Плюс авто-бэкап раз в сутки в `/data/backups/` (хранятся 10 шт.) |
| Обновить код вручную | `bash scripts/update_server.sh` (БД на томе сохраняется) |
| Перезапуск | `docker compose restart` |
| Остановить | `docker compose down` (том с БД **не** трогается; `down -v` — НЕ делать, удалит БД) |

**Данные** живут на именованном томе `rqm-data` (`/data/rqm.db`) — переживают
перезапуск, редеплой и пересборку контейнера.

## Рабочий процесс разработки и авто-деплой

Рабочая схема:

1. Локальная разработка идёт в ветке `developer`.
2. Локально проверяешь изменения: `python -m pytest -q` и/или
   `docker compose up -d --build`.
3. Когда версия готова, вливаешь изменения в `master` и пушишь:
   `git checkout master && git merge developer && git push origin master`.
4. GitHub Actions запускает `.github/workflows/deploy.yml` и прогоняет job
   `Test`.
5. Production-сервер сам раз в минуту проверяет `origin/master` через systemd
   timer `rqm-auto-deploy.timer`. Если появился новый commit, сервер выполняет
   `bash scripts/update_server.sh`.
6. Серверный скрипт подтягивает `origin/master`, пересобирает контейнер,
   перезапускает бота и прогоняет smoke-check. Если smoke-check не проходит,
   скрипт пытается откатить контейнер на предыдущий commit. `.env` и
   SQLite-данные остаются на сервере.

Проверить server-side авто-деплой:

```bash
systemctl status rqm-auto-deploy.timer
journalctl -u rqm-auto-deploy.service -n 100 --no-pager
```

На сервере клон должен быть подключён к GitHub-репозиторию и иметь доступ к
`origin/master`. В рабочем дереве на сервере не редактируй код вручную:
`scripts/update_server.sh` специально приводит код к точному состоянию
`origin/master`.

В GitHub включи branch protection для `master`: required status check `Test`.
Шпаргалка лежит в `.github/BRANCH_PROTECTION.md`.

Smoke/backup проверки:
```bash
python -m scripts.check_backup /path/to/backup.db
docker compose exec -T rqm-nav python -m app.smoke
docker compose exec -T rqm-nav python -m app.smoke --network
```

---

## Безопасность (уже учтено в коде)

- Bot/Telegraph токены **не пишутся** в логи/консоль (httpx заглушён + редактор
  секретов).
- Runtime-контейнер чинит владельца `/data`, затем запускает бота не от root.
  В compose включён `no-new-privileges`, а capabilities сведены к минимуму для
  `chown`/сброса прав на старых volume.
- Пользователям показываются общие сообщения об ошибках, без внутренностей.
- Лимит 10 запросов/мин на пользователя; очередь и кулдаун на скачивания;
  длина поиска ограничена. Админы — без лимитов.
- SQL только через параметры (инъекции невозможны), весь пользовательский текст
  экранируется.

**Что сделать на хосте:**
- `chmod 600 .env` (сделано выше) — чтобы токен не читался другими.
- `scripts/update_server.sh` предупреждает, если `.env` читается группой/всеми,
  но лучше сразу держать права `600`.
- **Не открывай порт 8080** наружу в Oracle (Security List / iptables). Он нужен
  только для healthcheck'а и отдаёт лишь `{"ok":true}`; публично выставлять не
  надо. Для polling-бота входящие порты вообще не требуются.
- Никому не показывай содержимое `.env` и `/data/rqm.db` (там токен Telegraph).

---

## Частые проблемы

- **`Could not reach api.telegram.org`** — регион/сеть блокирует Telegram.
  Убедись, что VM в не-РФ регионе. На крайний случай — `TELEGRAM_PROXY` в `.env`.
- **Бот не видит посты канала** — он не админ канала, либо неверный
  `CHANNEL_CHAT_ID`.
- **Навигация не появилась** — проверь, что у поста есть хэштег, привязанный к
  проекту, и ссылка-глава. Неизвестные хэштеги по умолчанию не публикуются:
  они попадают в conflict и ждут привязки в админке.
