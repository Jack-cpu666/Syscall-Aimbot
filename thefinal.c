// thefinals.c - AIRPLANE Defense System Implementation

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <onnxruntime_c_api.h>

// Constants and Definitions
#define FLIGHT_SPEED 0.6f
#define VIEW_RADIUS 300
#define RELIABILITY_THRESHOLD 0.50f
#define AUTO_ENGAGE false

#define MEM_COMMIT 0x00001000
#define MEM_RESERVE 0x00002000
#define MEM_RELEASE 0x00008000
#define PAGE_EXECUTE_READWRITE 0x40
#define INPUT_MOUSE 0
#define MOUSEEVENTF_MOVE 0x0001
#define MOUSEEVENTF_LEFTDOWN 0x0002
#define MOUSEEVENTF_LEFTUP 0x0004

#define VK_LBUTTON 0x01
#define VK_RBUTTON 0x02
#define VK_F1 0x70
#define VK_F2 0x71

#define CONSOLE_COLOR_DEFAULT 7
#define CONSOLE_COLOR_GREEN 10
#define CONSOLE_COLOR_RED 12
#define CONSOLE_COLOR_YELLOW 14
#define CONSOLE_COLOR_CYAN 11
#define CONSOLE_COLOR_MAGENTA 13

#define MODEL_PATH L"C:\\Users\\Security\\Desktop\\Security Files\\Eshan VOR\\new AIRPLANE defincesystem\\best.onnx"
#define INPUT_TENSOR_DIMS {1, 3, 640, 640}
#define NUM_BOXES 8400
#define OUTPUT_DIM 84

typedef struct _CUSTOM_MOUSE_INPUT {
    LONG dx;
    LONG dy;
    DWORD mouseData;
    DWORD dwFlags;
    DWORD time;
    ULONG_PTR dwExtraInfo;
} CUSTOM_MOUSE_INPUT;

typedef union _CUSTOM_INPUT_UNION {
    CUSTOM_MOUSE_INPUT mi;
} CUSTOM_INPUT_UNION;

typedef struct _CUSTOM_INPUT {
    DWORD type;
    CUSTOM_INPUT_UNION u;
} CUSTOM_INPUT;

typedef SHORT(*GetKeyStateFunc)(INT);
typedef UINT(*SendInputFunc)(UINT, CUSTOM_INPUT*, INT);

typedef struct {
    void* addr;
} MemoryBlock;

typedef struct {
    float x1, y1, x2, y2, score;
} DetectionBox;

struct FlightControlSystem {
    OrtEnv* env;
    OrtSession* session;
    char* input_name;
    char** output_names;
    size_t num_outputs;
    bool use_gpu;
    int view_size;
    int screen_w;
    int screen_h;
    int center_x;
    int center_y;
    bool system_active;
    double fps;
    double frame_times[30];
    int frame_index;
    int frame_count;
    double last_frame_time;
    double fps_update_time;
    int last_object_count;
    GetKeyStateFunc GetKeyStateInternal;
    SendInputFunc SendInputInternal;
    MemoryBlock* mem_blocks;
    int num_mem_blocks;
    HDC screen_dc;
    HDC mem_dc;
    HBITMAP bmp;
    RECT capture_rect;
};

const OrtApi* g_ort = NULL;

// Utility Functions
void SetConsoleColor(WORD color) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
}

void PrintColored(const char* text, WORD color) {
    SetConsoleColor(color);
    printf("%s", text);
    SetConsoleColor(CONSOLE_COLOR_DEFAULT);
}

OrtStatus* CheckStatus(OrtStatus* status, const OrtApi* ort) {
    if (status != NULL) {
        const char* msg = ort->GetErrorMessage(status);
        fprintf(stderr, "ERROR: %s\n", msg);
        ort->ReleaseStatus(status);
    }
    return status;
}

uint32_t GetDispatchId(const char* func_name) {
    HMODULE h_mod = LoadLibraryA("win32u.dll");
    if (!h_mod) return 0;
    FARPROC proc = GetProcAddress(h_mod, func_name);
    if (!proc) {
        FreeLibrary(h_mod);
        return 0;
    }
    uint32_t id = *(uint32_t*)((uintptr_t)proc + 4);
    FreeLibrary(h_mod);
    return id;
}

void* CreateDispatchWrapper(uint32_t id, size_t* stub_size) {
    uint8_t stub[] = { 0x4C, 0x8B, 0xD1, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x05, 0xC3 };
    memcpy(&stub[4], &id, sizeof(uint32_t));
    *stub_size = sizeof(stub);
    void* mem = VirtualAlloc(NULL, *stub_size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if (mem) {
        memcpy(mem, stub, *stub_size);
    }
    return mem;
}

void AddMemoryBlock(struct FlightControlSystem* fcs, void* addr) {
    fcs->mem_blocks = realloc(fcs->mem_blocks, (fcs->num_mem_blocks + 1) * sizeof(MemoryBlock));
    fcs->mem_blocks[fcs->num_mem_blocks].addr = addr;
    fcs->num_mem_blocks++;
}

void InitializeInternalFunctions(struct FlightControlSystem* fcs) {
    uint32_t id_key = GetDispatchId("NtUserGetAsyncKeyState");
    uint32_t id_input = GetDispatchId("NtUserSendInput");
    if (!id_key || !id_input) {
        fprintf(stderr, "Failed to resolve dispatch IDs\n");
        exit(1);
    }
    size_t size;
    void* key_mem = CreateDispatchWrapper(id_key, &size);
    void* input_mem = CreateDispatchWrapper(id_input, &size);
    if (!key_mem || !input_mem) {
        fprintf(stderr, "Failed to create dispatch wrappers\n");
        exit(1);
    }
    AddMemoryBlock(fcs, key_mem);
    AddMemoryBlock(fcs, input_mem);
    fcs->GetKeyStateInternal = (GetKeyStateFunc)key_mem;
    fcs->SendInputInternal = (SendInputFunc)input_mem;
}

void InitializeModel(struct FlightControlSystem* fcs, const OrtApi* ort, OrtAllocator* allocator) {
    OrtStatus* status;
    OrtSessionOptions* options = NULL;
    status = ort->CreateSessionOptions(&options);
    if (CheckStatus(status, ort) != NULL) exit(1);

    // Try CUDA first (for NVIDIA GPU)
    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(cuda_options));
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.do_copy_in_default_stream = 1;

    status = ort->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
    fcs->use_gpu = (CheckStatus(status, ort) == NULL);

    if (!fcs->use_gpu) {
        PrintColored("CUDA not available, using CPU\n", CONSOLE_COLOR_YELLOW);
    }

    // Optimize options
    ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
    ort->SetIntraOpNumThreads(options, 4);

    status = ort->CreateSession(fcs->env, MODEL_PATH, options, &fcs->session);
    if (CheckStatus(status, ort) != NULL) {
        fprintf(stderr, "Failed to create session\n");
        exit(1);
    }
    ort->ReleaseSessionOptions(options);

    // Get input name
    size_t num_inputs;
    ort->SessionGetInputCount(fcs->session, &num_inputs);
    if (num_inputs < 1) exit(1);
    status = ort->SessionGetInputName(fcs->session, 0, allocator, &fcs->input_name);
    if (CheckStatus(status, ort) != NULL) exit(1);

    // Get output names
    ort->SessionGetOutputCount(fcs->session, &fcs->num_outputs);
    fcs->output_names = malloc(fcs->num_outputs * sizeof(char*));
    for (size_t i = 0; i < fcs->num_outputs; i++) {
        status = ort->SessionGetOutputName(fcs->session, i, allocator, &fcs->output_names[i]);
        if (CheckStatus(status, ort) != NULL) exit(1);
    }

    PrintColored("Model initialized successfully\n", CONSOLE_COLOR_GREEN);
    PrintColored(fcs->use_gpu ? "GPU acceleration enabled\n" : "Using CPU execution\n", CONSOLE_COLOR_CYAN);
}

void CaptureScreenRegion(struct FlightControlSystem* fcs, uint8_t** rgb_data, int* width, int* height) {
    *width = fcs->capture_rect.right - fcs->capture_rect.left;
    *height = fcs->capture_rect.bottom - fcs->capture_rect.top;

    SelectObject(fcs->mem_dc, fcs->bmp);
    BitBlt(fcs->mem_dc, 0, 0, *width, *height, fcs->screen_dc, fcs->capture_rect.left, fcs->capture_rect.top, SRCCOPY);

    BITMAPINFOHEADER bi = { 0 };
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = *width;
    bi.biHeight = -*height; // Top-down
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    uint8_t* bgra_data = malloc((*height) * (*width) * 4);
    if (!bgra_data) exit(1);

    GetDIBits(fcs->mem_dc, fcs->bmp, 0, *height, bgra_data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    *rgb_data = malloc((*height) * (*width) * 3);
    if (!*rgb_data) exit(1);

    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            int idx4 = (y * (*width) + x) * 4;
            int idx3 = (y * (*width) + x) * 3;
            (*rgb_data)[idx3 + 0] = bgra_data[idx4 + 2]; // R
            (*rgb_data)[idx3 + 1] = bgra_data[idx4 + 1]; // G
            (*rgb_data)[idx3 + 2] = bgra_data[idx4 + 0]; // B
        }
    }
    free(bgra_data);
}

void ResizeNearest(uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h) {
    float scale_x = (float)dst_w / src_w;
    float scale_y = (float)dst_h / src_h;
    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            int sx = (int)(dx / scale_x);
            int sy = (int)(dy / scale_y);
            sx = max(0, min(sx, src_w - 1));
            sy = max(0, min(sy, src_h - 1));
            int src_idx = (sy * src_w + sx) * 3;
            int dst_idx = (dy * dst_w + dx) * 3;
            dst[dst_idx + 0] = src[src_idx + 0];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }
}

float* PreprocessImage(uint8_t* rgb_data, int w, int h) {
    uint8_t* resized = malloc(640 * 640 * 3);
    if (!resized) exit(1);
    ResizeNearest(rgb_data, w, h, resized, 640, 640);

    float* input = malloc(3 * 640 * 640 * sizeof(float));
    if (!input) exit(1);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                int hwc_idx = (y * 640 + x) * 3 + c;
                int chw_idx = c * 640 * 640 + y * 640 + x;
                input[chw_idx] = (float)resized[hwc_idx] / 255.0f;
            }
        }
    }
    free(resized);
    return input;
}

float ComputeIoU(const DetectionBox* a, const DetectionBox* b) {
    float inter_x = max(0.0f, min(a->x2, b->x2) - max(a->x1, b->x1));
    float inter_y = max(0.0f, min(a->y2, b->y2) - max(a->y1, b->y1));
    float inter_area = inter_x * inter_y;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

int* NonMaxSuppression(DetectionBox* boxes, int num_boxes, float iou_thresh, int* num_kept) {
    int* indices = malloc(num_boxes * sizeof(int));
    for (int i = 0; i < num_boxes; i++) indices[i] = i;

    // Sort by score descending (simple bubble sort, since small n)
    for (int i = 0; i < num_boxes - 1; i++) {
        for (int j = i + 1; j < num_boxes; j++) {
            if (boxes[indices[i]].score < boxes[indices[j]].score) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    bool* suppressed = calloc(num_boxes, sizeof(bool));
    int* kept = malloc(num_boxes * sizeof(int));
    *num_kept = 0;

    for (int i = 0; i < num_boxes; i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        kept[(*num_kept)++] = idx;
        for (int j = i + 1; j < num_boxes; j++) {
            int jdx = indices[j];
            if (suppressed[jdx]) continue;
            if (ComputeIoU(&boxes[idx], &boxes[jdx]) > iou_thresh) {
                suppressed[jdx] = true;
            }
        }
    }

    free(indices);
    free(suppressed);
    return kept;
}

DetectionBox* PostprocessOutput(float* output_data, int orig_w, int orig_h, float conf_thresh, float iou_thresh, int* num_dets) {
    DetectionBox* boxes = malloc(NUM_BOXES * sizeof(DetectionBox));
    *num_dets = 0;

    for (int k = 0; k < NUM_BOXES; k++) {
        float score = output_data[4 * NUM_BOXES + k];
        if (score < conf_thresh) continue;

        float cx = output_data[0 * NUM_BOXES + k];
        float cy = output_data[1 * NUM_BOXES + k];
        float w = output_data[2 * NUM_BOXES + k];
        float h = output_data[3 * NUM_BOXES + k];

        float scale_x = (float)orig_w / 640.0f;
        float scale_y = (float)orig_h / 640.0f;

        float half_w = w * scale_x * 0.5f;
        float half_h = h * scale_y * 0.5f;
        cx *= scale_x;
        cy *= scale_y;

        boxes[*num_dets].x1 = cx - half_w;
        boxes[*num_dets].y1 = cy - half_h;
        boxes[*num_dets].x2 = cx + half_w;
        boxes[*num_dets].y2 = cy + half_h;
        boxes[*num_dets].score = score;
        (*num_dets)++;
    }

    if (*num_dets == 0) {
        free(boxes);
        return NULL;
    }

    int num_kept;
    int* kept = NonMaxSuppression(boxes, *num_dets, iou_thresh, &num_kept);

    DetectionBox* final_boxes = malloc(num_kept * sizeof(DetectionBox));
    for (int i = 0; i < num_kept; i++) {
        final_boxes[i] = boxes[kept[i]];
    }
    *num_dets = num_kept;

    free(kept);
    free(boxes);
    return final_boxes;
}

void CalculateFPS(struct FlightControlSystem* fcs) {
    LARGE_INTEGER freq, current;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&current);
    double current_time = (double)current.QuadPart / freq.QuadPart;

    double frame_time = current_time - fcs->last_frame_time;
    fcs->last_frame_time = current_time;

    fcs->frame_times[fcs->frame_index] = frame_time;
    fcs->frame_index = (fcs->frame_index + 1) % 30;
    fcs->frame_count++;

    if (current_time - fcs->fps_update_time >= 0.5) {
        double sum = 0.0;
        for (int i = 0; i < 30; i++) sum += fcs->frame_times[i];
        double avg_time = sum / 30.0;
        fcs->fps = 1.0 / avg_time;
        fcs->fps_update_time = current_time;
    }
}

void UpdateStatusLine(struct FlightControlSystem* fcs) {
    WORD fps_color = (fcs->fps >= 30) ? CONSOLE_COLOR_GREEN : (fcs->fps >= 20) ? CONSOLE_COLOR_YELLOW : CONSOLE_COLOR_RED;
    char fps_text[32];
    sprintf(fps_text, "FPS: %.1f", fcs->fps);

    char targets_text[32];
    sprintf(targets_text, "Objects: %d", fcs->last_object_count);

    char mouse_status[32] = "";
    bool left_down = fcs->GetKeyStateInternal(VK_LBUTTON) < 0;
    bool right_down = fcs->GetKeyStateInternal(VK_RBUTTON) < 0;
    if (left_down && right_down) {
        strcpy(mouse_status, " | L+R CONTROL");
    }
    else if (left_down) {
        strcpy(mouse_status, " | LEFT CONTROL");
    }
    else if (right_down) {
        strcpy(mouse_status, " | RIGHT CONTROL");
    }

    printf("\033[K"); // Clear line if ANSI supported
    PrintColored("[!] SYSTEM [", CONSOLE_COLOR_DEFAULT);
    PrintColored(fcs->system_active ? "ACTIVE" : "INACTIVE", fcs->system_active ? CONSOLE_COLOR_GREEN : CONSOLE_COLOR_RED);
    PrintColored("] | ", CONSOLE_COLOR_DEFAULT);
    PrintColored(fps_text, fps_color);
    PrintColored(" | ", CONSOLE_COLOR_DEFAULT);
    PrintColored(targets_text, CONSOLE_COLOR_CYAN);
    PrintColored(mouse_status, CONSOLE_COLOR_MAGENTA);
    printf("\r");
    fflush(stdout);
}

bool IsSystemActive(struct FlightControlSystem* fcs) {
    return fcs->system_active;
}

bool IsEngaging(struct FlightControlSystem* fcs) {
    return (fcs->GetKeyStateInternal(VK_LBUTTON) < 0) || (fcs->GetKeyStateInternal(VK_RBUTTON) < 0);
}

bool IsFiring(struct FlightControlSystem* fcs) {
    return fcs->GetKeyStateInternal(VK_LBUTTON) < 0;
}

bool IsAligned(int x, int y, int center_x, int center_y) {
    int thresh = 5;
    return abs(x - center_x) <= thresh && abs(y - center_y) <= thresh;
}

void AdjustDirection(struct FlightControlSystem* fcs, int x, int y) {
    if (!IsEngaging(fcs)) return;

    float dx = (x - fcs->center_x) * FLIGHT_SPEED;
    float dy = (y - fcs->center_y) * FLIGHT_SPEED;

    if (fabs(dx) > 0.5f || fabs(dy) > 0.5f) {
        CUSTOM_INPUT input = { 0 };
        input.type = INPUT_MOUSE;
        input.u.mi.dx = (LONG)dx;
        input.u.mi.dy = (LONG)dy;
        input.u.mi.dwFlags = MOUSEEVENTF_MOVE;
        fcs->SendInputInternal(1, &input, sizeof(CUSTOM_INPUT));
    }
}

void SimulateEngage(struct FlightControlSystem* fcs) {
    CUSTOM_INPUT down = { 0 };
    down.type = INPUT_MOUSE;
    down.u.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    fcs->SendInputInternal(1, &down, sizeof(CUSTOM_INPUT));

    Sleep(10);

    CUSTOM_INPUT up = { 0 };
    up.type = INPUT_MOUSE;
    up.u.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    fcs->SendInputInternal(1, &up, sizeof(CUSTOM_INPUT));
}

void RunSystemLoop(struct FlightControlSystem* fcs) {
    const OrtApi* ort = g_ort;
    OrtAllocator* allocator;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    static bool prev_f1 = false;
    static bool prev_f2 = false;

    while (true) {
        CalculateFPS(fcs);
        UpdateStatusLine(fcs);

        bool curr_f1 = fcs->GetKeyStateInternal(VK_F1) < 0;
        bool curr_f2 = fcs->GetKeyStateInternal(VK_F2) < 0;

        if (!prev_f1 && curr_f1) {
            fcs->system_active = !fcs->system_active;
        }
        if (!prev_f2 && curr_f2) {
            break;
        }
        prev_f1 = curr_f1;
        prev_f2 = curr_f2;

        uint8_t* rgb_data;
        int cap_w, cap_h;
        CaptureScreenRegion(fcs, &rgb_data, &cap_w, &cap_h);

        float* input_data = PreprocessImage(rgb_data, cap_w, cap_h);
        free(rgb_data);

        OrtValue* input_tensor = NULL;
        OrtMemoryInfo* mem_info = NULL;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
        int64_t input_shape[] = INPUT_TENSOR_DIMS;
        ort->CreateTensorWithDataAsOrtValue(mem_info, input_data, 3 * 640 * 640 * sizeof(float), input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
        ort->ReleaseMemoryInfo(mem_info);

        const char* input_names[] = { fcs->input_name };
        OrtValue* inputs[] = { input_tensor };
        OrtValue** outputs = malloc(fcs->num_outputs * sizeof(OrtValue*));
        for (size_t i = 0; i < fcs->num_outputs; i++) outputs[i] = NULL;

        ort->Run(fcs->session, NULL, input_names, (const OrtValue* const*)inputs, 1, (const char* const*)fcs->output_names, fcs->num_outputs, outputs);

        float* output_data;
        ort->GetTensorMutableData(outputs[0], (void**)&output_data);

        int num_dets;
        DetectionBox* dets = PostprocessOutput(output_data, cap_w, cap_h, RELIABILITY_THRESHOLD, 0.45f, &num_dets);
        fcs->last_object_count = num_dets;

        if (num_dets > 0) {
            float min_dist = INFINITY;
            DetectionBox* closest = NULL;
            for (int i = 0; i < num_dets; i++) {
                DetectionBox* box = &dets[i];
                int x1 = (int)box->x1;
                int y1 = (int)box->y1;
                int x2 = (int)box->x2;
                int y2 = (int)box->y2;

                bool is_self = (x1 < 15) || (x1 < fcs->view_size / 5 && y2 > fcs->view_size / 1.2f);
                if (is_self) continue;

                int center_X = (x1 + x2) / 2;
                int box_h = y2 - y1;
                int head_Y = y1 + (int)(box_h * 0.25f);

                float dist = sqrtf(powf(center_X - fcs->view_size / 2.0f, 2) + powf(head_Y - fcs->view_size / 2.0f, 2));
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = box;
                    box->x1 = (float)center_X; // Temp store for target
                    box->y1 = (float)head_Y;
                }
            }

            if (closest) {
                int target_x = (int)closest->x1 + fcs->capture_rect.left;
                int target_y = (int)closest->y1 + fcs->capture_rect.top;

                if (IsAligned(target_x, target_y, fcs->center_x, fcs->center_y)) {
                    if (AUTO_ENGAGE && !IsFiring(fcs)) {
                        SimulateEngage(fcs);
                    }
                }

                if (IsSystemActive(fcs)) {
                    AdjustDirection(fcs, target_x, target_y);
                }
            }
        }

        free(dets);
        free(input_data);
        ort->ReleaseValue(input_tensor);
        for (size_t i = 0; i < fcs->num_outputs; i++) {
            ort->ReleaseValue(outputs[i]);
        }
        free(outputs);
    }
}

void CleanupSystem(struct FlightControlSystem* fcs, const OrtApi* ort, OrtAllocator* allocator) {
    ort->ReleaseSession(fcs->session);
    ort->ReleaseEnv(fcs->env);
    for (size_t i = 0; i < fcs->num_outputs; i++) {
        allocator->Free(allocator, fcs->output_names[i]);
    }
    free(fcs->output_names);
    allocator->Free(allocator, fcs->input_name);

    for (int i = 0; i < fcs->num_mem_blocks; i++) {
        VirtualFree(fcs->mem_blocks[i].addr, 0, MEM_RELEASE);
    }
    free(fcs->mem_blocks);

    DeleteObject(fcs->bmp);
    DeleteDC(fcs->mem_dc);
    ReleaseDC(NULL, fcs->screen_dc);
}

int main() {
    struct FlightControlSystem fcs = { 0 };
    g_ort = OrtGetApiBase()->GetApi(17);  // Use API version 17 for compatibility with ORT 1.17.1
    const OrtApi* ort = g_ort;
    OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "system", &fcs.env);
    if (CheckStatus(status, ort) != NULL) return 1;

    OrtAllocator* allocator;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    fcs.view_size = VIEW_RADIUS;
    fcs.screen_w = GetSystemMetrics(SM_CXSCREEN);
    fcs.screen_h = GetSystemMetrics(SM_CYSCREEN);
    fcs.center_x = fcs.screen_w / 2;
    fcs.center_y = fcs.screen_h / 2;
    fcs.system_active = true;
    fcs.fps = 0.0;
    memset(fcs.frame_times, 0, sizeof(fcs.frame_times));
    fcs.frame_index = 0;
    fcs.frame_count = 0;
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    fcs.last_frame_time = (double)now.QuadPart / freq.QuadPart;
    fcs.fps_update_time = fcs.last_frame_time;
    fcs.last_object_count = 0;

    fcs.capture_rect.left = fcs.center_x - fcs.view_size / 2;
    fcs.capture_rect.top = fcs.center_y - fcs.view_size / 2;
    fcs.capture_rect.right = fcs.capture_rect.left + fcs.view_size;
    fcs.capture_rect.bottom = fcs.capture_rect.top + fcs.view_size;

    fcs.screen_dc = GetDC(NULL);
    fcs.mem_dc = CreateCompatibleDC(fcs.screen_dc);
    fcs.bmp = CreateCompatibleBitmap(fcs.screen_dc, fcs.view_size, fcs.view_size);

    InitializeInternalFunctions(&fcs);
    InitializeModel(&fcs, ort, allocator);

    PrintColored("System ready. Press F1 to toggle, F2 to exit.\n", CONSOLE_COLOR_GREEN);
    PrintColored("Engage with left or right control.\n", CONSOLE_COLOR_CYAN);

    RunSystemLoop(&fcs);

    PrintColored("\nExiting system...\n", CONSOLE_COLOR_YELLOW);
    CleanupSystem(&fcs, ort, allocator);

    return 0;
}