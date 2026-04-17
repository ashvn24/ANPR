from onvif import ONVIFCamera
cam = ONVIFCamera('192.168.68.102', 80, 'admin', '123456')
media = cam.create_media_service()
profiles = media.GetProfiles()
uri = media.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}, 'ProfileToken': profiles[0].token})
print(uri.Uri)
