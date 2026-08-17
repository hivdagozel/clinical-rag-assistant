import urllib.request
import re

def main():
    url = "https://www.titck.gov.tr/dinamikmodul/43"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    import ssl
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=context) as response:
            html = response.read().decode('utf-8')
            # HTML içindeki xlsx uzantılı tüm linkleri bul
            links = re.findall(r'href="(https?://[^"]+\.xlsx)"', html)
            print("Bulunan Excel Bağlantıları:")
            for l in links[:5]:
                print(f" - {l}")
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
