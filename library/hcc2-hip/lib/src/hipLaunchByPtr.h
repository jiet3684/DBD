
#define EXPORT __attribute__((visibility("default")))

#define HIP_FATBIN_CODE  0xBA55ED50
#define HIP_MAX_STREAMS  24
#define HIP_MAX_KERNARGS 1024
#define HIP_MAXARGSTRUCTSZ 1024
#define HIP_MAX_KERNELS 128
// VALCHECK is xdeadbeef
#define VALCHECK 3735928559

struct hipi_fbwrapper {
  unsigned int magic;
  unsigned int version;
  void*        binary;
  void*        dummy1;
};
typedef struct hipi_fbwrapper hipi_fbwrapper;

struct  hipi_fbheader {
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};
typedef struct hipi_fbheader hipi_fbheader;

struct  hipi_partheader{
  unsigned short         type;
  unsigned short         dummy1;
  unsigned int           headerSize;
  unsigned long long int partSize;
  unsigned long long int dummy2;
  unsigned int           dummy3;
  unsigned int           subarch;
};

typedef struct hipi_partheader hipi_partheader;
typedef unsigned hipErr_t;
typedef struct {char bytes[64];}  HIP_IpcMemHandle_t;
typedef struct {char bytes[64];}  HIP_IpcEventHandle_t;
typedef struct {char bytes[20];}  HIP_ChannelFormatDesc;
typedef struct {char bytes[648];} HIP_DeviceProp;
typedef struct {char bytes[32];} HIP_PointerAttributes;

struct hipi_kernel_s {
  void*                    khaddr;
  uint64_t                 kernel_object;
  const char*              kernel_name;
  char*                    host_name;
  unsigned int             thread_limit;
  uint3                  * tid;
  uint3                  * bid;
  dim3                   * bDim;
  dim3                   * gDim;
  int                    * wSize;
};
typedef struct hipi_kernel_s hipi_kernel_t;

struct hipi_launchdata_s{
  dim3                    blockDim;
  dim3                    gridDim;
  long long                smsize;
  hipStream_t           * stream;
  unsigned int argstructsize;
  char argstruct[HIP_MAXARGSTRUCTSZ];
};
typedef struct hipi_launchdata_s hipi_launchdata_t;

struct hipi_global_s{
  unsigned int             kernel_count;
  hipDevice_t              hip_device;
  hipCtx_t                 hip_context;
  hipModule_t              hip_module;
  unsigned int             syncq_len;
  char                     agent_name[64];
  unsigned int             streamq_len[HIP_MAX_STREAMS];
};
typedef struct hipi_global_s hipi_global_t;

/* Code kinds supported by the driver */
typedef enum {
  HIP_PART_PTX      = 0x0001,
  HIP_PART_ELF      = 0x0002,
  HIP_PART_OLDCUBIN = 0x0004,
  HIP_PART_ELFGCN   = 0x0008
} hipi_part_t;

typedef unsigned int hipi_err_t;
/* Values for a hipi_err_t */
typedef enum {
  HIPBYPTR_SUCCESS = 0,
  HIPBYPTR_FAIL    = 1
} hipbyptr_errs;

enum DevicePropOffset {
  name =                             0,
  totalGlobalMem =                   256,
  sharedMemPerBlock =                264,
  regsPerBlock =                     272,
  warpSize =                         276,
  memPitch =                         280,
  maxThreadsPerBlock =               288,
  maxThreadsDim =                    292,
  maxGridSize =                      304,
  clockRate =                        316,
  totalConstMem =                    320,
  major =                            328,
  minor =                            332,
  textureAlignment =                 336,
  texturePitchAlignment =            344,
  deviceOverlap =                    352,
  multiProcessorCount =              356,
  kernelExecTimeoutEnabled =         360,
  integrated =                       364,
  canMapHostMemory =                 368,
  computeMode =                      372,
  maxTexture1D =                     376,
  maxTexture1DMipmap =               380,
  maxTexture1DLinear =               384,
  maxTexture2D =                     388,
  maxTexture2DMipmap =               396,
  maxTexture2DLinear =               404,
  maxTexture2DGather =               416,
  maxTexture3D =                     424,
  maxTexture3DAlt =                  436,
  maxTextureCubemap =                448,
  maxTexture1DLayered =              452,
  maxTexture2DLayered =              460,
  maxTextureCubemapLayered =         472,
  maxSurface1D =                     480,
  maxSurface2D =                     484,
  maxSurface3D =                     492,
  maxSurface1DLayered =              504,
  maxSurface2DLayered =              512,
  maxSurfaceCubemap =                524,
  maxSurfaceCubemapLayered =         528,
  surfaceAlignment =                 536,
  concurrentKernels =                544,
  ECCEnabled =                       548,
  pciBusID =                         552,
  pciDeviceID =                      556,
  pciDomainID =                      560,
  tccDriver =                        564,
  asyncEngineCount =                 568,
  unifiedAddressing =                572,
  memoryClockRate =                  576,
  memoryBusWidth =                   580,
  l2CacheSize =                      584,
  maxThreadsPerMultiProcessor =      588,
  streamPrioritiesSupported =        592,
  globalL1CacheSupported =           596,
  localL1CacheSupported =            600,
  sharedMemPerMultiprocessor =       608,
  regsPerMultiprocessor =            616,
  managedMemory =                    620,
  isMultiGpuBoard =                  624,
  multiGpuBoardGroupID =             628,
  hostNativeAtomicSupported =        632,
  singleToDoublePrecisionPerfRatio = 636,
  pageableMemoryAccess =             640,
  concurrentManagedAccess =          644
};

