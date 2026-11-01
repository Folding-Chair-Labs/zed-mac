// Minimal IOKit UVC control-request shim: sends class requests to a device's VideoControl interface (pipe 0)
// without claiming it, so it works while AVFoundation/UVCAssistant streams. No root needed.
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/usb/IOUSBLib.h>

static void setnum(CFMutableDictionaryRef d, CFStringRef k, int v){
  CFNumberRef n=CFNumberCreate(NULL,kCFNumberIntType,&v); CFDictionarySetValue(d,k,n); CFRelease(n);}

void *uvc_open(int vid, int pid, int ifnum){
  CFMutableDictionaryRef m=IOServiceMatching("IOUSBHostInterface");
  CFMutableDictionaryRef pm=CFDictionaryCreateMutable(NULL,0,&kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
  setnum(pm,CFSTR("idVendor"),vid); setnum(pm,CFSTR("idProduct"),pid); setnum(pm,CFSTR("bInterfaceNumber"),ifnum);
  CFDictionarySetValue(m,CFSTR(kIOPropertyMatchKey),pm); CFRelease(pm);
  io_service_t svc=IOServiceGetMatchingService(kIOMainPortDefault,m);
  if(!svc) return NULL;
  IOCFPlugInInterface **plug=NULL; SInt32 score;
  kern_return_t kr=IOCreatePlugInInterfaceForService(svc,kIOUSBInterfaceUserClientTypeID,kIOCFPlugInInterfaceID,&plug,&score);
  IOObjectRelease(svc);
  if(kr||!plug) return NULL;
  IOUSBInterfaceInterface **intf=NULL;
  (*plug)->QueryInterface(plug,CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID),(LPVOID*)&intf);
  (*plug)->Release(plug);
  return intf;
}

// returns IOReturn (0 = success); *done = bytes transferred
int uvc_request(void *h, int bmRequestType, int bRequest, int wValue, int wIndex, void *buf, int len, int *done){
  IOUSBInterfaceInterface **intf=h;
  IOUSBDevRequestTO q={.bmRequestType=bmRequestType,.bRequest=bRequest,.wValue=wValue,.wIndex=wIndex,
                       .wLength=len,.pData=buf,.noDataTimeout=1000,.completionTimeout=2000};
  IOReturn r=(*intf)->ControlRequestTO(intf,0,&q);
  if(done) *done=q.wLenDone;
  return r;
}

void uvc_close(void *h){ IOUSBInterfaceInterface **intf=h; if(intf) (*intf)->Release(intf); }
