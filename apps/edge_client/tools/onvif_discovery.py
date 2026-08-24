import socket
import uuid
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

ONVIF_USER = "admin"
ONVIF_PASSWORD = "password"

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)

DISCOVERY_MESSAGE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope
    xmlns:e="http://www.w3.org/2003/05/soap-envelope"
    xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{message_id}</w:MessageID>
    <w:To e:mustUnderstand="true">
      urn:schemas-xmlsoap-org:ws:2005:04:discovery
    </w:To>
    <w:Action e:mustUnderstand="true">
      http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe
    </w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>
"""

GET_CAPABILITIES = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope
    xmlns:s="http://www.w3.org/2003/05/soap-envelope"
    xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Body>
    <tds:GetCapabilities>
      <tds:Category>All</tds:Category>
    </tds:GetCapabilities>
  </s:Body>
</s:Envelope>
"""

GET_PROFILES = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope
    xmlns:s="http://www.w3.org/2003/05/soap-envelope"
    xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
  <s:Body>
    <trt:GetProfiles/>
  </s:Body>
</s:Envelope>
"""


def soap_post(url, body):
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8"
    }

    response = requests.post(
        url,
        data=body.encode("utf-8"),
        headers=headers,
        auth=HTTPDigestAuth(ONVIF_USER, ONVIF_PASSWORD),
        timeout=5,
    )

    response.raise_for_status()
    return response.text


def discover_onvif(timeout=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)

    message = DISCOVERY_MESSAGE.format(
        message_id=str(uuid.uuid4())
    ).encode("utf-8")

    sock.sendto(message, WS_DISCOVERY_ADDR)

    devices = set()

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            break

        xml = data.decode("utf-8", errors="ignore")

        try:
            root = ET.fromstring(xml)

            for elem in root.iter():
                if elem.tag.endswith("XAddrs") and elem.text:
                    for xaddr in elem.text.split():
                        devices.add(xaddr)

        except ET.ParseError:
            pass

    sock.close()
    return sorted(devices)


def get_media_service(device_url):
    xml = soap_post(device_url, GET_CAPABILITIES)
    root = ET.fromstring(xml)

    for elem in root.iter():
        if elem.tag.endswith("Media"):
            for child in elem:
                if child.tag.endswith("XAddr"):
                    return child.text

    return None


def get_profiles(media_url):
    xml = soap_post(media_url, GET_PROFILES)
    root = ET.fromstring(xml)

    profiles = []

    for elem in root.iter():
        if elem.tag.endswith("Profiles"):
            token = elem.attrib.get("token")

            name = None
            for child in elem:
                if child.tag.endswith("Name"):
                    name = child.text
                    break

            if token:
                profiles.append({
                    "token": token,
                    "name": name,
                })

    return profiles


def get_stream_uri(media_url, profile_token):
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope
    xmlns:s="http://www.w3.org/2003/05/soap-envelope"
    xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
    xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <trt:GetStreamUri>
      <trt:StreamSetup>
        <tt:Stream>RTP-Unicast</tt:Stream>
        <tt:Transport>
          <tt:Protocol>RTSP</tt:Protocol>
        </tt:Transport>
      </trt:StreamSetup>
      <trt:ProfileToken>{profile_token}</trt:ProfileToken>
    </trt:GetStreamUri>
  </s:Body>
</s:Envelope>
"""

    xml = soap_post(media_url, body)
    root = ET.fromstring(xml)

    for elem in root.iter():
        if elem.tag.endswith("Uri"):
            return elem.text

    return None


def add_rtsp_credentials(uri, username, password):
    if not uri:
        return None

    if not uri.startswith("rtsp://"):
        return uri

    if "@" in uri:
        return uri

    return uri.replace(
        "rtsp://",
        f"rtsp://{username}:{password}@",
        1,
    )


def main():
    print("Searching ONVIF cameras...")

    devices = discover_onvif()

    if not devices:
        print("No ONVIF devices found.")
        return

    print(f"Found {len(devices)} device(s)\n")

    for device_url in devices:
        print("=" * 70)
        print("Device:", device_url)

        try:
            media_url = get_media_service(device_url)

            if not media_url:
                print("Media service not found.")
                continue

            print("Media service:", media_url)

            profiles = get_profiles(media_url)

            if not profiles:
                print("No media profiles found.")
                continue

            for profile in profiles:
                print()
                print("Profile name:", profile["name"])
                print("Profile token:", profile["token"])

                rtsp_uri = get_stream_uri(
                    media_url,
                    profile["token"],
                )

                print("RTSP URI:", rtsp_uri)

                rtsp_with_auth = add_rtsp_credentials(
                    rtsp_uri,
                    ONVIF_USER,
                    ONVIF_PASSWORD,
                )

                print("RTSP with auth:", rtsp_with_auth)

        except requests.exceptions.RequestException as e:
            print("HTTP/ONVIF error:", e)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()