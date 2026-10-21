# План: поддержка ROCm + текстурирование (Mac и AMD)

Документ описывает доработку `hunyuan3d-2.1-mac` до мультибэкендного порта
(MPS / ROCm / CUDA / CPU) и включение PBR-текстурирования на обоих
не-CUDA бэкендах.

Ссылки на строки апстрима даны по `Tencent-Hunyuan/Hunyuan3D-2.1@main`.

---

## 1. Что реально мешает текстурированию (ревизия апстрима)

README сейчас утверждает, что paint-пайплайн требует xformers и
CUDA-компилированного дифференцируемого рендерера. Это не подтверждается кодом.

| Компонент | Реальная зависимость от CUDA | Вывод |
|---|---|---|
| `hy3dpaint/custom_rasterizer` | `rasterizer_gpu.cu` — единственный CUDA-файл. `rasterizer.cpp` содержит **полную CPU-реализацию** и диспетчеризацию по устройству тензора (`rasterize_image`: `if (device_id == -1) → rasterize_image_cpu`) | CPU-путь уже есть; ROCm собирается через hipify |
| `hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp` | Нет. Чистый pybind11 + STL, без torch, без CUDA, без OpenMP | Собирается везде |
| `hy3dpaint/DifferentiableRenderer/MeshRender.py` | `device="cuda"` только как **значение по умолчанию** конструктора; есть `.to(device)`. Растеризация вызывается из одной точки — `raster_rasterize` (MeshRender.py:509) | Параметризуемо |
| `xformers` / `flash-attn` / `bitsandbytes` | Не импортируются нигде в Python-коде | Блокера нет |
| `cupy-cuda12x`, `deepspeed` | Перечислены в `requirements.txt`, но в коде инференса не используются (`deepspeed` — только тренировка) | Мусор в requirements |
| RealESRGAN (`realesrgan` + `basicsr`) | Обычный torch. `RealESRGANer(half=True, gpu_id=None)` внутри выбирает `cuda if available else cpu` | ROCm — работает как есть; MPS — уходит на CPU |
| `bpy` (Blender) | `DifferentiableRenderer/mesh_utils.py:17` импортирует `bpy` на уровне модуля → `MeshRender.save_mesh` (MeshRender.py:653) падает без Blender | Нужна bpy-free замена |

Полный список жёстко прошитых устройств в paint-пайплайне:

- `hy3dpaint/textureGenPipeline.py:39` — `self.device = "cuda"`
- `hy3dpaint/textureGenPipeline.py:87` — `torch.cuda.empty_cache()`
- `hy3dpaint/textureGenPipeline.py:77` — `MeshRender(...)` создаётся **без** `device=`
- `hy3dpaint/DifferentiableRenderer/MeshRender.py:337` — `device="cuda"`
- `hy3dpaint/hunyuanpaintpbr/pipeline.py:245` — `.to("cuda")` в `convert_pil_list_to_tensor`
- `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py:750` — `.to("cuda:0")` безусловно, на пути инференса (`SelfAttnProcessor2_0.__call__`)
- `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py:690` — `cuda:0`/`cuda:1`, но только под `multiple_devices=True` (не наш путь)

**Асимметрия, определяющая объём работ:** PyTorch на ROCm представляется как CUDA
(`torch.cuda.is_available() == True`, устройство называется `cuda`, `torch.version.hip`
заполнен, `torch.version.cuda is None`). Поэтому **для ROCm ни одна из строк выше
не требует патча** — они работают как есть. Все патчи устройств нужны только для MPS.

Итого: текстурирование на Mac — не «невозможно», а «медленно». Растеризатор
имеет CPU-путь, инпейнт вообще CPU, а диффузия (основное время) идёт на MPS.

---

## 2. Целевая архитектура

### 2.1. `backend.py` (новый модуль — ядро изменений)

Заменяет `pick_device()` (gradio_app.py:178), `DTYPE` (gradio_app.py:186) и
`_install_mps_patches()` (gradio_app.py:44).

```python
# backend.py
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Backend:
    kind: str          # "rocm" | "cuda" | "mps" | "cpu"
    device: str        # "cuda" | "mps" | "cpu"   (для ROCm — именно "cuda")
    dtype: torch.dtype
    raster: str        # "gpu" | "cpu"  — бэкенд custom_rasterizer
    needs_device_patches: bool   # True только для mps/cpu
    autocast_ok: bool
    gpu_name: str

def detect(force_kind: str | None = None, force_raster: str | None = None) -> Backend:
    if torch.cuda.is_available():
        kind = "rocm" if torch.version.hip else "cuda"
        return Backend(kind, "cuda", torch.float16, force_raster or "gpu",
                       False, True, torch.cuda.get_device_name(0))
    if torch.backends.mps.is_available():
        return Backend("mps", "mps", torch.float16, force_raster or "cpu",
                       True, False, "Apple Silicon (MPS)")
    return Backend("cpu", "cpu", torch.float32, "cpu", True, False, "CPU")

def empty_cache(b: Backend) -> None: ...
def gpu_mem_info(b: Backend) -> tuple[int, int]: ...   # для UI и для выбора пресета
```

Переопределение через env: `HY3D_BACKEND`, `HY3D_RASTER`, `HY3D_DTYPE` —
нужно для отладки (в частности для бисекции проблемы искажённых текстур на ROCm, п. 6).

### 2.2. `compat_patches.py`

Текущие патчи (`gradio_app.py:44-90`) выносятся сюда и применяются
**условно** — `if backend.needs_device_patches`. На ROCm не применяется ничего:
подмена `torch.autocast(device_type="cuda")` на no-op там не только не нужна,
но и вредна (потеря fp16-autocast).

Добавляется `apply_paint_patches(backend)` — monkey-patch paint-пайплайна
(детали в п. 4.2).

### 2.3. Файловая структура

```
hunyuan3d-2.1-mac-rocm/
├── backend.py              ← новый: детект устройства и политики
├── compat_patches.py       ← новый: device-патчи (mps/cpu) + paint-патчи
├── paint.py                ← новый: обёртка Hunyuan3DPaintPipeline
├── mesh_io.py              ← новый: bpy-free save_mesh / OBJ+MTL / GLB PBR
├── gradio_app.py           ← бэкенд-агностичный UI + вкладка «Текстура»
├── lowpoly.py              ← без изменений
├── scripts/
│   ├── common.sh           ← общая логика install/fix
│   ├── install_mac.sh
│   ├── install_rocm.sh
│   ├── install_rocm_windows.ps1
│   ├── build_extensions.py ← сборка custom_rasterizer + mesh_inpaint_processor
│   └── fix.sh
├── patches/
│   ├── rasterizer_h.patch          ← HOSTDEV-макрос, условный include
│   ├── rasterizer_cpu_only.patch   ← #ifdef CR_CPU_ONLY в dispatch
│   └── setup_cpu.py                ← CppExtension-вариант сборки
├── requirements_mac.txt
├── requirements_rocm.txt   ← новый
└── docs/ROCM_TEXTURING_PLAN.md
```

`launch.sh` / `launch.ps1` выставляют переменные окружения по бэкенду:

| Бэкенд | Переменные |
|---|---|
| MPS | `PYTORCH_ENABLE_MPS_FALLBACK=1`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` |
| ROCm | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `MIOPEN_FIND_MODE=FAST`, `MIOPEN_USER_DB_PATH` (кэш ядер — иначе первый прогон очень долгий), `HSA_OVERRIDE_GFX_VERSION` **только** если сборка не содержит нативных ядер под GPU |

---

## 3. Установка на ROCm

### 3.1. Linux (рекомендуемый путь, полный GPU)

```bash
# torch под ROCm
pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
    torch torchvision torchaudio
```

Для **gfx1151** (Radeon 8060S / Ryzen AI Max+ 395) официальные колёса не годятся —
эта архитектура не в матрице поддержки, а wheels с pytorch.org собраны под ROCm 6.x
и на ROCm 7.2 падают. Нужны nightly TheRock:

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    torch torchvision torchaudio
```

`install_rocm.sh` должен:
1. Определить GPU: `rocminfo | grep gfx` → `GFX_ARCH`.
2. По `GFX_ARCH` выбрать индекс колёс (таблица: gfx1100/1101/1102 → stable; gfx1151, gfx1200/1201 → nightlies).
3. Проверить `torch.version.hip` и `torch.cuda.is_available()`, вывести имя GPU.
4. Поставить `requirements_rocm.txt`.
5. Собрать расширения (`scripts/build_extensions.py`).
6. Скачать веса: shape + `hunyuan3d-paintpbr-v2-1/*` + RealESRGAN + `facebook/dinov2-giant`.

### 3.2. Windows-native ROCm

PyTorch на ROCm для Windows существует как public preview (ROCm 7.1.1+/7.2.x)
и покрывает Radeon 7000/9000 и часть Ryzen AI 300 / AI Max.

**Проверено на железе (см. п. 8): нативные расширения на Windows недоступны вообще.**
Не только HIP-ядро, но и CPU-вариант: torch-колесо не линкует ни одного
C++-расширения. Поэтому на Windows используется `raster="torch"` —
чистый PyTorch, без компилятора.

Деталь, которая остаётся актуальной для CPU-сборок на других платформах:
в `rasterizer.cpp` используется `z_min.data_ptr<long>()`, а `long` в MSVC
32-битный, а в macOS `int64_t` — это `long long`, то есть `data_ptr<long>`
там вообще не инстанцируется. Для обеих платформ надо брать сорцы из
`custom_rasterizer/lib/custom_rasterizer_kernel_for_windows/` (всё переведено
на `int64_t`/`uint64_t`). Обратно: эти сорцы нельзя использовать для GPU-сборки
на Linux, потому что `atomicMin(uint64_t*)` там разрешается в `unsigned long*`
и не совпадает с перегрузкой CUDA/HIP для `unsigned long long*`.

### 3.3. WSL2

Работает, ROCm в WSL поддерживается для части карт, но для gfx1151 путь мутный.
Как вариант — не первый приоритет.

---

## 4. Текстурирование

### 4.1. Сборка нативных расширений (`scripts/build_extensions.py`)

**`custom_rasterizer`:**

- ROCm / CUDA: `pip install -e hy3dpaint/custom_rasterizer` **без изменений**.
  `torch.utils.cpp_extension.CUDAExtension` при `torch.version.hip` сам прогоняет
  hipify по `.cu`, `#include <ATen/cuda/CUDAContext.h>` → HIP-аналог,
  `at::cuda::getCurrentCUDAStream()` → masquerading-версия, `<<<>>>` поддерживается HIP,
  `atomicMin`/`atomicExch` на `unsigned long long` в HIP есть.
  Подтверждение работоспособности: сторонний порт Hunyuan3D-2 для AMD/Linux
  раздаёт готовые wheels `custom_rasterizer`, собранные под ROCm 6.4.1.

- Mac / CPU-only: собирать `patches/setup_cpu.py` — `CppExtension` только из
  `rasterizer.cpp` + `grid_neighbor.cpp` (в `grid_neighbor.cpp` CUDA нет вообще),
  с `-DCR_CPU_ONLY`. Два микропатча:

  `rasterizer.h`:
  ```cpp
  #if defined(__CUDACC__) || defined(__HIPCC__)
  #include <ATen/cuda/CUDAContext.h>
  #define HOSTDEV __host__ __device__
  #else
  #define HOSTDEV
  #endif
  // далее __host__ __device__ → HOSTDEV в трёх inline-функциях
  ```

  `rasterizer.cpp` (иначе линковка требует `rasterize_image_gpu`):
  ```cpp
  std::vector<torch::Tensor> rasterize_image(...) {
  #ifdef CR_CPU_ONLY
      return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
  #else
      int device_id = V.get_device();
      return device_id == -1 ? rasterize_image_cpu(...) : rasterize_image_gpu(...);
  #endif
  }
  ```

**`mesh_inpaint_processor`:** заменить `compile_mesh_painter.sh` на кроссплатформенный
шаг. Апстримная команда на macOS не слинкуется без `-undefined dynamic_lookup`:

```bash
# macOS
c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup \
  $(python -m pybind11 --includes) mesh_inpaint_processor.cpp \
  -o mesh_inpaint_processor$(python3-config --extension-suffix)
```

На Windows — через маленький `setup.py` с `pybind11.setup_helpers.Pybind11Extension`.

### 4.2. `compat_patches.apply_paint_patches(backend)`

Все патчи — monkey-patch над импортированными модулями, апстрим не правим
(это соблюдает текущее правило проекта: «upstream fixes — upstream»).

1. **Устройство конфига**
   ```python
   Hunyuan3DPaintConfig.__init__  →  self.device = backend.device
   ```
   плюс `render_size` / `texture_size` / `max_selected_view_num` из пресета памяти.

2. **`MeshRender` получает device**
   `Hunyuan3DPaintPipeline.__init__` (textureGenPipeline.py:77) создаёт `MeshRender`
   без `device=`. Патчим — передаём `backend.device`.

3. **Развязка растеризатора и остального рендера** — центральный патч.
   `MeshRender.raster_rasterize` (MeshRender.py:509) обёртывается:
   ```python
   def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
       if RASTER == "cpu":
           dev = pos.device
           findices, barycentric = _orig_rasterize(self, pos.cpu(), tri.cpu(), resolution)
           return findices.to(dev), barycentric.to(dev)
       return _orig_rasterize(self, pos, tri, resolution, ranges, grad_db)
   ```
   Смысл: только сама растеризация уходит на CPU, а вся остальная тензорная
   математика рендера (интерполяция UV на 2048², бэйкинг в 4096²) остаётся на GPU.
   `raster_interpolate` (MeshRender.py:545) — чистый torch, device-агностичен.

4. **fp16 → CPU.** CPU-путь растеризатора требует `float32` для `V` и `int32` для `F`.
   `raster_rasterize` уже приводит `float64→float32` и `int64→int32`, но не `float16`.
   Добавить приведение `half → float32` перед CPU-вызовом.

5. **Только MPS:** `.to("cuda")` в `hunyuanpaintpbr/pipeline.py:245` и `.to("cuda:0")`
   в `attn_processor.py:750` → `backend.device`. Для ROCm не патчим.

6. **`torch.cuda.empty_cache()`** (textureGenPipeline.py:87) → `backend.empty_cache`.

7. **RealESRGAN.** `imageSuperNet` создаёт `RealESRGANer(half=True, gpu_id=None)`.
   - ROCm: работает как есть.
   - MPS: `RealESRGANer` уйдёт на CPU, а `half=True` на CPU катастрофически медленно →
     форсировать `half=False`, либо (лучше) заменить на прямой прогон `RRDBNet`
     на `backend.device`. Апскейл ×4 от 512 до 2048 на MPS — секунды.
   - Не забыть `torchvision_fix.apply_fix()` до импорта `basicsr`
     (`basicsr` тянет удалённый `torchvision.transforms.functional_tensor`).

8. **bpy-free сохранение (`mesh_io.py`).** `MeshRender.save_mesh` (MeshRender.py:653)
   вызывает `save_mesh` из `mesh_utils`, который недоступен без Blender.
   Патчим `MeshRender.save_mesh` на свою реализацию: OBJ + MTL + `albedo.png` /
   `metallic_roughness.png` / `normal.png`, затем GLB через апстримный
   `hy3dpaint/convert_utils.py::create_glb_with_pbr_materials` (pygltflib, без bpy).
   `convert_obj_to_glb` (mesh_utils.py:260) не использовать — он на bpy.

### 4.3. `paint.py` — обёртка

```python
def load_paint_pipeline(backend, preset) -> Hunyuan3DPaintPipeline
def texture_mesh(mesh_path, image, backend, preset, progress_cb) -> dict
    # → {"obj": ..., "glb": ..., "albedo": PIL, "mr": PIL, "views": [PIL, ...]}
```
Ленивая загрузка, как `_load_pipeline()` сейчас. Обязательно —
**выгрузка shape-пайплайна перед загрузкой paint** (см. п. 5).

### 4.4. UI

- Новая секция «Текстурирование (PBR)» под превью:
  чекбокс «Генерировать текстуру», `max_num_view` (6–9), `resolution` (512 / 768),
  селектор растеризатора (`auto` / `gpu` / `cpu`), пресет памяти.
- Второе превью `gr.Model3D` для текстурированного GLB + галерея
  albedo / metallic-roughness / multiview-ракурсов.
- Форматы скачивания: GLB (с PBR-материалами) и ZIP (OBJ + MTL + карты).
  Текущие `DOWNLOAD_FORMATS` (gradio_app.py:783) для текстур не годятся —
  STL/OFF/3MF текстуры не несут; их надо помечать как «геометрия без текстуры».
- Текстурирование применяется к **оптимизированному** мешу, а не к `original`
  (paint внутри сам делает `remesh_mesh` до 40k треугольников —
  `utils/simplify_mesh_utils.py`, поэтому lowpoly-пресеты и paint нужно
  согласовать: при включённом текстурировании имеет смысл `use_remesh=False`,
  если пользователь уже прогнал свой decimation).
- Переименовать «MPS vs CPU» → «GPU vs CPU» во всех трёх языках; `DeviceUsageTracker`
  (gradio_app.py:103) уже device-агностичен, менять только i18n-ключи и подписи.
- Строки `description` в i18n (gradio_app.py:216, 313, и ru-блок) содержат
  «Apple Silicon», «MPS», «Texturing needs CUDA and is disabled» — вынести
  бэкенд в подстановку `{backend}` и убрать утверждение про невозможность текстур.

---

## 5. Память — оказалась НЕ ограничением (проверено на железе)

Изначальная оценка была основана на `Win32_ComputerSystem.TotalPhysicalMemory`
= 31.6 ГБ и `AdapterRAM` = 4 ГБ. Оба числа обманывают.

Фактический замер: `torch.cuda.get_device_properties(0).total_memory` =
**107.9 GiB**. У машины 128 ГБ, из которых крупный кусок отдан GPU как VRAM-карвеаут,
поэтому Windows и не показывает его в «физической памяти» (31.6 + ~96 ≈ 128).

Следствие: заявленных апстримом ~21 ГБ достаточно с большим запасом,
`suggest_paint_preset()` корректно выбирает `normal`. Меры ниже остаются
полезными, но перестают быть критическими:

1. **Variable Graphics Memory** в AMD Software / UEFI — актуально только для
   систем с малым карвеаутом. `scripts/doctor.py` печатает фактически видимый GPU
   объём, чтобы это было видно сразу, а не после OOM.
2. **Разделить стадии.** Shape и paint никогда не держать в памяти одновременно.
   Это остаётся обязательным независимо от объёма: реализовано как
   `unload_shape_pipeline()` перед загрузкой paint-модели. Именно на этом
   спотыкаются другие AMD-порты («даже на 32 ГБ падает, пришлось разделить
   на два скрипта»).
3. **Пресеты:**

   | Пресет | views | resolution | render_size | texture_size |
   |---|---|---|---|---|
   | Low (≤32 ГБ) | 6 | 512 | 1024 | 2048 |
   | Normal (48 ГБ+) | 8 | 768 | 2048 | 4096 |

   Дефолты апстрима — `render_size=2048`, `texture_size=4096`
   (textureGenPipeline.py:49-50).
4. `pipeline.enable_model_cpu_offload()` / `enable_sequential_cpu_offload()`
   для multiview-диффузии как аварийный режим.
5. `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` против фрагментации.

---

## 6. Риски и их отработка

| Риск | Оценка | Что делаем |
|---|---|---|
| **Искажённые текстуры на ROCm.** Открытый баг ROCm/ROCm#5981: Hunyuan3D-2.1 на ROCm 6.4–7.2 (r9700 / gfx1201) даёт корректный меш и битые текстуры; на NVIDIA не воспроизводится, первопричина не установлена | Высокий — прямо по нашей задаче | Селектор растеризатора даёт бисекцию: `raster=cpu` исключает HIP-ядро и атомики из подозреваемых. Если при `raster=cpu` текстуры чистые → баг в `rasterizer_gpu.cu` под HIP (вероятно `atomicMin` на 64-битном z-буфере или квантование глубины). Если битые и там → fp16-путь attention/VAE, проверять `HY3D_DTYPE=float32` и отключение SDPA-бэкендов. В обоих случаях у нас есть рабочая конфигурация |
| gfx1151 вне официальной матрицы поддержки ROCm | Высокий | Только nightly-колёса `rocm.nightlies.amd.com/v2/gfx1151/`; installer проверяет `torch.version.hip` и делает smoke-тест матмула перед установкой остального |
| 31.6 ГБ общей памяти на пайплайн, рассчитанный на 21 ГБ VRAM | Средний | Пресет Low + разделение стадий (п. 5) |
| Сборка HIP-расширения на Windows-native ROCm | Средний | Обходится `raster=cpu`; полный GPU-путь — на Linux |
| `std::min`/`std::max` в `__device__`-коде `rasterizer_gpu.cu` под HIP-clang | Низкий | Они `constexpr`, компилируются; если нет — заменить на `fminf`/`fmaxf` в патче |
| `basicsr` ломается на современном torchvision | Низкий | `torchvision_fix.apply_fix()` до импорта |
| Скорость CPU-растеризации | Низкий | Барицентрический проход — `O(width*height)` однопоточно на view; на 1024² это доли секунды против минут диффузии |

---

## 7. Порядок работ и точки проверки

Каждый этап заканчивается проверяемым результатом.

1. **Рефакторинг бэкенда.** `backend.py` + `compat_patches.py`, `gradio_app.py`
   переведён на них. i18n обезличен.
   *Проверка:* на Mac поведение не изменилось (shape-генерация как была).

2. **Установка ROCm.** `install_rocm.sh`, `requirements_rocm.txt`, `launch.sh`
   с ROCm-переменными.
   *Проверка:* `torch.version.hip` заполнен, имя GPU выводится, матмул на GPU считается.

3. **Shape-генерация на ROCm.** Никакие device-патчи не применяются.
   *Проверка:* меш из тестового изображения; трекер показывает ≈100 % на `cuda`.
   Сравнить с эталоном с Mac — геометрия должна совпадать в пределах шума seed.

4. **Сборка расширений.** `build_extensions.py`: GPU-вариант на ROCm,
   CPU-вариант на Mac/Windows.
   *Проверка:* `import custom_rasterizer, mesh_inpaint_processor`;
   юнит-тест: один треугольник → сравнить `findices`/`barycentric`
   между CPU- и GPU-путём на ROCm (это же и тест на баг из п. 6).

5. **Текстурирование на ROCm.** `paint.py`, `mesh_io.py`, paint-патчи.
   *Проверка:* `assets/case_1` (image.png + mesh.glb из апстрима) → текстурированный
   GLB, визуально сравнить с апстримным эталоном. Прогнать оба значения `raster`.

6. **Текстурирование на Mac.** Тот же путь с `raster=cpu`.
   *Проверка:* тот же кейс; замер времени по стадиям (multiview / super-res / bake / inpaint).

7. **UI и документация.** Вкладка текстурирования, README (две платформы,
   таблица «что работает где»), `NOTICE` — дополнить список модификаций
   (патчи растеризатора, bpy-free сохранение, device-патчи), `LICENSE`-замечание:
   paint-модель и её выводы — под той же Tencent-лицензией.

---

## 8. Что подтвердилось и что изменилось при реализации

Всё ниже проверено на реальной машине (Ryzen AI Max+ 395 / Radeon 8060S,
gfx1151, Windows 11, torch 2.9.1+rocmsdk20260116, HIP 7.2), а не выведено из чтения кода.

### Подтвердилось

- **hipify чист на этих исходниках.** `CUDAExtension` под ROCm перевёл
  `rasterizer_gpu.cu` с результатом «unsupported CUDA function calls: 0,
  replaced kernel launches: 3». То есть на Linux нативное HIP-ядро действительно
  собирается без правок исходников.
- **`mesh_inpaint_processor` собирается на Windows** и устанавливается рядом с
  `MeshRender`. Он не трогает torch, поэтому от проблем torch-расширений не зависит.
- **`grid_neighbor.cpp` — мёртвый код на пути инференса.** `build_hierarchy` и
  `build_hierarchy_with_feat` только объявлены и забиндены; ни одна строка Python
  в апстриме их не вызывает. Исключены из CPU-сборки.
- **На ROCm патчи устройств не нужны.** `compat_patches.apply()` — осознанный no-op:
  проверено, что `torch.autocast(device_type="cuda")` работает штатно и fp16 сохраняется.

### Изменилось

1. **На Windows нельзя собрать НИ ОДНО torch-расширение.** Это не про
   `custom_rasterizer` — минимальный пустой `PYBIND11_MODULE` над
   `<torch/extension.h>` падает так же: `LNK2001`, неэкспортированный из `c10.lib`
   конструктор `c10::ValueError(SourceLocation, std::string)`. Проверено тремя
   отдельными пробами (пустой модуль, `torch::zeros`, `data_ptr<float>`).
   Дополнительно: `_rocm_sdk_devel` разворачивает только `bin/`, а `include/`,
   `lib/`, `cmake/` остаются пустыми — HIP-хедеров и оффлайн-компилятора нет.

   → Появился третий режим растеризатора: **`torch_rasterizer.py`, чистый PyTorch**.
   Компилятор не нужен вообще, работает на том устройстве, где лежат тензоры.
   Сверен с построчной транскрипцией C++-циклов
   (`tests/test_torch_rasterizer.py`, 14 проверок на CPU и GPU): `findices`
   совпадают побитово, барицентрики до 4e-6. Замер: 80k граней при 2048² —
   82 мс на GPU, 292 мс при 4096². Против минут диффузии это не горячая точка.

   Это заодно закрывает и Mac лучше, чем исходный план: там растеризация тоже
   уходит на GPU, а не на CPU.

2. **`mesh_io.py` не переписывает запись OBJ.** Выяснилось, что `save_obj_mesh`
   в `mesh_utils` — чистый numpy+cv2, Blender ему не нужен; недоступен он только
   из-за `import bpy` на уровне модуля. Поэтому достаточно **заглушки `bpy`**,
   а апстримный писатель OBJ/MTL/текстур используется как есть.

   Важнее другое: `MeshRender` импортирует из `mesh_utils` не только `save_mesh`,
   но и `load_mesh`. Без заглушки падает **первый** шаг пайплайна
   (`render.load_mesh`), а не сохранение. В исходном плане это было недооценено.

   GLB пишется своим экспортёром через trimesh `PBRMaterial`, а не апстримным
   `create_glb_with_pbr_materials`: тот пишет `temp.glb` и `mr_combined.png`
   в текущий рабочий каталог и помечает все встроенные изображения как
   `image/png`, хотя текстуры записаны в JPEG.

3. **Патчи `.to("cuda")` пришлось делать на уровне `Tensor.to`/`Module.to`.**
   Оба проблемных вызова недостижимы по отдельности: один — в замыкании внутри
   300-строчного `__call__`, второй — безусловная строка в
   `SelfAttnProcessor2_0.__call__`. Редирект сужен до строковых форм
   `"cuda"`/`"cuda:N"` и ставится только когда устройство не CUDA-подобное,
   то есть на ROCm и CUDA этой цены нет вообще.

4. **`snapshot_download` пришлось перехватить.** `multiviewDiffusionNet` жёстко
   вызывает его с `repo_id`, поэтому просто указать локальный путь в конфиге
   недостаточно — иначе он заново качает ~10 ГБ мимо `weights/`.

5. **`--raster` стал четырёхзначным** (`auto|gpu|cpu|torch`), а сборщик
   деградирует по цепочке GPU → CPU → torch с внятным диагнозом на каждом шаге
   вместо стены вывода линковщика.

### Найдено при фактической установке и прогоне

Всё ниже вылезло только при реальном запуске и уже исправлено в коде.

1. **`sdp_kernel(enable_math=False)` ломает ROCm.** hy3dshape оборачивает attention
   в `torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
   enable_mem_efficient=True)`. На NVIDIA это осмысленная оптимизация, на ROCm ни
   flash, ни mem-efficient эти формы не покрывают, и PyTorch падает с
   `RuntimeError: No available kernel. Aborting execution.`
   → `compat_patches` теперь снимает это ограничение и на ROCm тоже (но не на CUDA).
   Это единственное исключение из правила «на ROCm не патчим ничего».

2. **Конфликт версий gradio ↔ transformers.** gradio 6 требует
   `huggingface_hub>=1.16`, transformers 4.x — `<1.0`. Взаимоисключающие.
   → `gradio>=5.0,<6` в requirements. Дополнительно: пакетная установка
   батчами ломает пины (последний батч поднял hub до 1.27), поэтому
   `requirements_rocm.txt` надо ставить **одной** командой.

3. **`hf download --include` принимает ровно один шаблон.** Несколько подряд
   уходят в позиционные FILENAMES, и часть весов молча не скачивается —
   именно так пропал DiT при полностью «успешном» выводе.
   → в Windows-установщике теперь по одному `--include` на шаблон.

4. **Symlink'и HF-кеша на Windows.** `snapshot_download` докачивает блобы и затем
   падает с `OSError: [WinError 1314] A required privilege is not held by the
   client`, оставляя нерабочий snapshot (нужен Developer Mode).
   → `backend.py` на Windows выставляет `HF_HUB_DISABLE_SYMLINKS=1`.

5. **setuptools не находит MSVC при запуске не из dev-консоли.** Сборка
   `mesh_inpaint_processor` падала с «Unable to find a compatible Visual Studio
   installation», хотя Build Tools установлены.
   → `build_extensions.py` сам находит `vcvars64.bat` (через vswhere плюс
   типовые пути) и повторяет сборку внутри него с `DISTUTILS_USE_SDK=1`.
   Через `cmd /c "..."` это не работает — cmd съедает внешние кавычки, поэтому
   используется временный `.bat`.

6. **Децимация в remesh не работала.** Апстрим зовёт
   `simplify_quadric_decimation(40000)` позиционно; в trimesh 4.x/5.x первый
   параметр — `percent`, то есть «4 000 000 %», и меш не упрощается вовсе.
   → `paint._patch_remesh()` передаёт `face_count=` по ключу.

7. **Зеркальная камера в превью.** Численные тесты растеризатора это поймать не
   могли: карты индексов и барицентрики совпадали идеально. Обнаружилось только
   на рендере — вдавленный в геометрию текст «HY3D» читался наоборот. Камера
   смотрит вдоль +Z, то есть находится в −Z, и оттуда мировое +X слева.
   → в `render_preview.py` негируются и X, и Y.

8. **Трекер операций замедляет генерацию примерно на порядок.**
   `TorchFunctionMode` перехватывает каждую операцию в Python. Первый прогон
   с `--track-gpu` не завершился за 25 минут, без него — 376 с.
   → предупреждение добавлено в подсказки UI на трёх языках.

### Наблюдение в апстриме, оставленное как есть

`textureGenPipeline.py:165` — `for i in range(len(enhance_images))`. Здесь
`enhance_images` это словарь с двумя ключами (`albedo`, `mr`), поэтому цикл
проходит только `i = 0, 1`, и к `render_size` приводятся лишь **первые два**
ракурса. Остальные остаются в разрешении после апскейла ×4 (при resolution=512
это 2048). Судя по имени переменной, задумано было `len(enhance_images["albedo"])`.

Порчи геометрии/UV это не даёт: `MeshRender.back_project` берёт разрешение из
самого изображения (`resolution = image.shape[:2]`), так что проекция остаётся
согласованной. Практический эффект — больше памяти и времени на запекание, чем
подразумевает `render_size`, то есть пресет «Низкий» ограничивает память слабее,
чем заявляет.

Намеренно **не** патчится: это не сбой, а лишняя детализация, и «исправление»
незаметно ухудшило бы качество относительно поведения апстрима. Если
текстурирование начнёт падать по памяти — вот первое, что надо поправить
(одна строка в `paint.py` рядом с остальными патчами).

### ФИНАЛ: текстурирование на ROCm работает. Потолок — 256px

Предыдущий раздел («упирается во внимание») был верен по механике, но
**неверен по выводу**. Настоящей причиной падений оказалась моя собственная
переменная `MIOPEN_FIND_MODE=FAST`: она направляет MIOpen в
`GetSolutionsFallback`, который передаёт солверу нулевой рабочий буфер
(`workspace required: 150994944, provided ptr: 0000000000000000 size: 0`) и
затем его разыменовывает. Свёрточный paint-пайплайн умирал так за полторы
минуты. После её удаления предупреждения исчезли, и текстурирование прошло.

Замеренная таблица (Radeon 8060S, gfx1151, Windows, ROCm 7.2):

| Ракурсы | Разрешение | Матрица внимания | Результат |
|---|---|---|---|
| 4 | 256 px | 1.9 ГиБ | ✅ 517 с (холодный кеш ядер) |
| 6 | **256 px** | **4.2 ГиБ** | ✅ **136 с, корректный PBR** |
| 6 | 320 px | 10.3 ГиБ | ❌ зависание машины |
| 6 | 512 px | 67.5 ГиБ | ❌ выделение невозможно, зависание |
| 8 | 768 px | 607.5 ГиБ | ❌ безнадёжно |

Отсюда пресет `safe` (6×256) и порог `_ATTN_SAFE_LIMIT = 5 ГиБ`: он пропускает
доказанную конфигурацию и отсекает 320px, которое машину роняет. Оба значения
взяты из замеров, а не назначены.

Ещё один пойманный дефект — **инверсия оси V при экспорте GLB**.
`MeshRender.get_mesh` отдаёт UV в конвенции OBJ (`vtx_uv[:,1] = 1 - v`,
MeshRender.py:841), где V считается снизу, а glTF считает сверху. Мой экспортёр
передавал их без преобразования. На хаотичной раскладке xatlas это выглядит не
как отражённая текстура, а как размазанный по модели цветовой шум — очень легко
принять за «испорченное запекание» (и, возможно, часть сообщений о битых
текстурах на ROCm — это оно). Исправлено в `mesh_io.save_textured_mesh`.

Отдельно: рендер превью в том же процессе **сразу после** paint-пайплайна один
раз дал почти чёрный кадр, тогда как тот же файл в свежем процессе рисовался
корректно. В `tests/e2e_visual.py` перед превью теперь выгружается paint-модель
и освобождается кеш памяти.

### Механика ограничения (предыстория)

Это главный вывод фактических прогонов, и он меняет оценку из п. 1.

Multiview-внимание paint-UNet сцепляет все ракурсы в одну последовательность,
поэтому матрица оценок растёт как **квадрат** (ракурсы × токены). Без
flash/mem-efficient SDPA PyTorch материализует её целиком:

| Ракурсы | Разрешение | Матрица внимания |
|---|---|---|
| 4 | 256 px | 1.9 ГиБ |
| 6 | 320 px | 10.3 ГиБ |
| 6 | 512 px | **67.5 ГиБ** |
| 8 | 768 px | **607.5 ГиБ** |

Коэффициент откалиброван по факту: замеренный запрос 67.5 ГиБ при 6×512 даёт
batch×heads = 60 (см. `paint.estimate_attention_bytes`).

Что перепробовано на gfx1151 / Windows / ROCm 7.2:

1. **Math-бэкенд как есть** — запрос одного тензора 67.5 ГиБ. Не выделяется даже
   при ~98 ГиБ свободных: упирается в лимит одиночного выделения.
2. **`expandable_segments`** — не помогает, запрос действительно на один тензор.
3. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — включает flash/mem-efficient,
   и **падает с SIGSEGV** (совпадает с известными багами gfx1151).
4. **Снижение разрешения до 320** — гигантское выделение уходит, прогон доходит
   до 217 с и падает уже на 3 ГиБ при 53 ГиБ занятых: остаётся кумулятивное
   давление math-внимания по слоям.
5. **Порционное внимание** (`compat_patches._patch_sdpa_chunking`, срез по оси
   запросов) — численно **точно** (проверено побитово в
   `tests/test_sdpa_chunking.py`, включая маски и fp16) и снимает отказ выделения.
   Но при бюджете 1 ГиБ это ~68 порций на вызов: прогон шёл >25 минут, а затем
   **сбросил графический драйвер и перезагрузил машину** — у APU память общая с
   системой, поэтому длительное давление роняет не процесс, а всю ОС.

Вывод: **текстурирование требует эффективного внимания**, и на этой конфигурации
его нет. Поэтому добавлена предполётная проверка `paint.attention_preflight()`,
которая по умолчанию **отказывает** с внятным объяснением вместо попытки
(обход — `HY3D_ALLOW_RISKY_TEXTURING=1`, с явным указанием, что цена ошибки —
перезагрузка машины). `suggest_paint_preset()` перестал предлагать «normal» вне
CUDA: объём пула там не является ограничением. `scripts/doctor.py` печатает
вердикт для конкретного железа.

Где текстурирование должно работать: Linux/ROCm на дискретной карте с рабочим
flash attention, либо NVIDIA. Это ещё не проверено — нет такого железа.

### Текущий статус этапов

| Этап | Статус |
|---|---|
| 1. Рефакторинг бэкенда (`backend.py`, `compat_patches.py`, i18n) | готово, проверено на железе |
| 2. Установка ROCm (Windows + Linux, `doctor.py`, лаунчеры) | готово; Windows-путь проверен частично (torch уже стоял) |
| 3. Shape-генерация на ROCm | **не проверено** — нужны веса (~15 ГБ) |
| 4. Сборка расширений | готово и проверено: `mesh_inpaint_processor` собран, растеризатор → torch |
| 5. Текстурирование на ROCm | код готов, **end-to-end не проверено** — нужны paint-веса (~10 ГБ) |
| 6. Текстурирование на Mac | **готово и проверено сквозным прогоном** на M4 Pro 24 ГБ — см. п. 10 |
| 7. UI и документация | готово; UI проверен статически (арность обработчиков, 40/40 outputs), вживую не поднят — gradio не установлен |

## 9. Что нужно решить до старта

1. **Целевая ОС для ROCm.** Linux (полный GPU-растеризатор, зрелый стек) или
   Windows-native preview (CPU-растеризатор, зато без второй ОС). От этого зависит
   `install_*`-скрипт и то, собираем ли мы HIP-ядро вообще.
2. **Оставляем ли Mac первичной платформой** (тогда репозиторий переименовывается
   и README перестраивается на две равные платформы) или ROCm — вторая цель
   в том же репозитории.
3. **Переименование репозитория.** Каталог уже называется `hunyuan3d-2.1-mac-rocm`,
   но `install.sh` клонирует и ссылается на `hunyuan3d-2.1-mac`.

## 10. MPS: что понадобилось на самом деле

Всё ниже проверено на MacBook с **M4 Pro, 24 ГБ unified memory, macOS 26.5,
torch 2.13.0** (`torch.mps.recommended_max_memory()` = 17.8 ГиБ).

Этап 6 плана («Текстурирование на Mac») формулировался как «тот же путь с
`raster=cpu`, проверить нечем». Проверка нашла три дефекта, ни один из которых
из чтения кода не следовал, и один — фатальный.

### Не подтвердилось (хорошие новости)

- **int64 на MPS работает.** Главный ожидаемый риск: `torch_rasterizer` пакует
  квантованную глубину и индекс треугольника в один int64-токен
  (`z * 2147483647 + tri`, начальное значение 4.6e18) и делает по нему
  `scatter_reduce_(reduce="amin")`. Metal исторически 64-битную арифметику
  поддерживает плохо. На torch 2.13 всё точно: `scatter_reduce_` по int64
  выполняется **нативно** (проверено с `PYTORCH_ENABLE_MPS_FALLBACK=0`), и
  `tests/test_torch_rasterizer.py` даёт побитовое совпадение `findices` с
  C++-оракулом на всех семи сценах.
- **`torch.Generator(device="mps")` создаётся.** `multiview_utils.forward_one`
  вызывает его безусловно; в старых torch это была ошибка, сейчас — нет.
- **fp16 SDPA на MPS считает корректно** (расхождение с CPU-fp32 — 7e-4).
- **Отказ выделения — это исключение, а не перезагрузка.** Одиночный буфер
  12 ГиБ выделяется, 24 ГиБ — нет, и Metal отвечает
  `RuntimeError: Invalid buffer size`. Цена ошибки на Mac качественно ниже,
  чем на APU AMD, поэтому предполётная проверка здесь мягче (см. ниже).

### Дефект 1 (фатальный): `torch.isin` в `back_project`

`MeshRender.back_project` (MeshRender.py:1275) в режиме `back_sample` делает
`torch.isin(valid_idx, inner_valid_idx)`. Оба аргумента — векторы индексов по
UV-атласу, на текстуре 1024² это сотни тысяч элементов. ATen выбирает между
широковещательным сравнением и сортировкой; на MPS выбирается
широковещательное, то есть промежуточный тензор `len(a) × len(b)`. Замер:
запрос **одного буфера на 111 ГБ**.

Отдельно неприятно, что это не исключение: отказ приходит из Metal, и процесс
умирает на `failed assertion 'Failed to allocate private MTLBuffer'` — поймать
`try/except` нельзя, лог обрывается на полуслове.

→ `compat_patches._patch_isin()`: сортировка `test_elements` один раз плюс
`searchsorted`. Тот же ответ, O(n log n), без большого промежуточного тензора,
и всё остаётся на GPU (200k × 700k — 90 мс). Ставится **только на MPS**: на
CUDA и ROCm ATen сам выбирает сортировку.

### Дефект 2: float64 по дороге на устройство

`MeshRender._convert_texture_format` (MeshRender.py:277) заканчивается на

```python
tex = np.array(tex) / 255.0            # float64
return torch.from_numpy(tex).to(device).float()
```

— сначала копирует на устройство, потом сужает. У Metal float64 нет вообще, и
это не краевой случай: пайплайн приходит сюда на каждом прогоне через
`set_texture_mr(texture_mr)` (textureGenPipeline.py:184) — то есть **в самом
конце**, когда весь бэйк уже оплачен.

→ `paint._patch_texture_conversion()` сужает до fp32 **до** копирования. Та же
арифметика в другом порядке; ставится только на MPS.

### Дефект 3: `timm` не было в `requirements_mac.txt`

`hy3dshape/models/denoisers/moe_layers.py` импортирует `timm`. Без него
shape-пайплайн падает на загрузке модели — `❌ Model load error: No module
named 'timm'`. Заодно в mac-файле не было вообще всех paint-зависимостей
(`realesrgan`, `basicsr`, `pymeshlab`, `fast_simplification`,
`pytorch-lightning`, `pybind11`) и верхних границ версий, которые удерживают
`transformers<5` / `huggingface_hub<1.0` / `gradio<6` совместимыми.

### Растеризатор: нативная CPU-сборка на macOS всё-таки собирается

План (п. 3.2) предполагал, что портируемые исходники
(`custom_rasterizer_kernel_for_windows`) годятся и для macOS. Фактически
clang их не берёт по двум причинам, обе исправлены в `build_extensions.py`:

1. `float vt[2] = {px + 0.5, py + 0.5};` — `int + 0.5` это double, а сужение
   double→float внутри списочной инициализации запрещено с C++11. MSVC и GCC
   пропускают с предупреждением, clang — ошибка. → `_patch_narrowing()`.
2. Собранный `.so` ссылается на `@rpath/libc10.dylib`, но rpath никто не
   прописывает: модуль компилируется, ставится и падает на импорте
   (`Library not loaded`). → сгенерированный `setup.py` добавляет
   `-Wl,-rpath,<torch>/lib`.

После этого нативный CPU-путь работает и **совпадает** с `torch_rasterizer`
(`findices` — точно, барицентрики — 1.6e-5). Замеры на 79k треугольников:

| Разрешение | нативный CPU | torch на CPU | torch на MPS |
|---|---|---|---|
| 1024² | 51.8 мс | 63.5 мс | 47.6 мс |
| 2048² | 165.1 мс | 237.5 мс | 151.6 мс |

Отсюда `backend._default_raster()` на MPS предпочитает `torch` даже когда
нативная сборка есть: она **CPU**-путь, то есть на каждый ракурс это копия
позиций вниз и карты индексов обратно, и выигрыша не даёт. Нативный путь
остаётся доступен через `HY3D_RASTER=cpu` — как независимая реализация для
сверки, если бэйк вызовет сомнения.

### Предполётная проверка внимания на MPS

Порог 5 ГиБ был выведен из замеров на gfx1151, где превышение роняло машину.
На Mac последствие другое, поэтому и порог другой: `paint.attention_limit()`
берёт 40 % от `recommended_max_memory()`, но не ниже прежних 5 ГиБ. На этой
машине это 7.1 ГиБ — пресет «Безопасный» (4.2 ГиБ) проходит, «Низкий»
(67.5 ГиБ) отклоняется с сообщением, которое называет цену честно: это
~17 последовательных порций на каждый вызов внимания, то есть очень долгий
прогон и своп, а не перезагрузка.

### Новый тест: `tests/test_paint_render.py`

Диффузионная половина paint-пайплайна требует 6.5 ГБ весов; вторая половина —
UV-развёртка, растеризация, back-projection, взвешенное запекание,
UV-инпейнт, экспорт OBJ/GLB — не требует ничего, и ломается на новом бэкенде
именно она. Тест гоняет реальный `MeshRender` на реальном меше с
`paint.apply_patches`, ровно как их ставит приложение, **и делает то же самое
на CPU**, после чего сверяет запечённые текстуры. Именно так и нашлись оба
первых дефекта.

Результат на MPS: 22/22, покрытие атласа 64.7 %, совпадение с CPU —
100.000 % текселей по маске и max 2.4e-7 по цвету.

### Сквозной прогон: работает

`tests/e2e_visual.py --steps 30 --octree 192 --tex-preset safe --tex-views 6
--tex-resolution 256` на `assets/demo.png`, M4 Pro 24 ГБ:

| Стадия | Время | Результат |
|---|---|---|
| import gradio_app (сборка UI-графа) | 1.2 с | ✅ |
| Генерация формы (DiT + VAE), 30 шагов, octree 192 | 344 с | ✅ 89 701 вершина, 179 404 грани |
| Превью необработанного меша (torch_rasterizer) | 0.5 с | ✅ |
| **Текстурирование PBR** (6 ракурсов / 256 px) | **512 с** | ✅ 40 000 граней, текстура 1024² |
| Превью текстурированного (torch_rasterizer) | 1.0 с | ✅ |

Диффузия формы — 7.6 с/шаг. Текстура визуально корректна: надпись «HY3D» на
табличке читается и цвета на месте, то есть ни инверсии UV, ни шума по атласу
(артефакт, который на ROCm принимали за «битое запекание») здесь нет.

Два замечания по итогам прогона:

- **`facebook/dinov2-giant` (~4.5 ГБ) не качал никто.** Апстрим тянет его
  лениво, через `AutoModel.from_pretrained` внутри `multiviewDiffusionNet.__init__`.
  На практике прогон молча стоял несколько минут посреди paint-стадии, ничего
  не печатая, уже после того как оплачена генерация формы. Теперь его качает
  `install.sh`, а `scripts/doctor.py` показывает предупреждение, если его нет
  в кеше HF.
- **24 ГБ — впритык.** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (его теперь ставит
  и `backend.py`, не только `launch.sh`) снимает мягкий лимит выделения, и
  прогон уходит в своп — на пике порядка 8 ГБ. Это осознанный размен: с
  включённым лимитом стадия падала бы с OOM после нескольких минут работы.
