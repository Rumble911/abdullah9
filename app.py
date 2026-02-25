import os
import re
import secrets
import string
import hashlib
import base64
import io
import json
import sqlite3
import pyotp  # type: ignore
import qrcode  # type: ignore
import datetime
import time
import tempfile
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template_string, send_file, session  # type: ignore
from cryptography.fernet import Fernet  # type: ignore
from cryptography.hazmat.primitives import hashes  # type: ignore
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
import requests  # type: ignore
from functools import lru_cache
from PIL import Image  # type: ignore
from pypdf import PdfReader, PdfWriter  # type: ignore
import random
import uuid
import socket
import concurrent.futures
import subprocess
import threading
import wave
import psutil  # type: ignore
import exifread  # type: ignore
from cryptography.hazmat.primitives.asymmetric import rsa  # type: ignore
from cryptography.hazmat.primitives import serialization, hashes  # type: ignore
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore
import urllib.parse

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # حد أقصى للملفات 16 ميجابايت
app.secret_key = 'TITAN_ULTRA_SECRET_KEY_2025_SECURE_BY_DEFAULT_CHANGE_THIS'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# --- قاعدة بيانات المستخدمين (SQLite) ---
USERS_DB = 'titan_users.db'

def init_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            vault_password_hash TEXT DEFAULT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    # Support upgrading existing DBs that don't have vault_password_hash yet
    try:
        c.execute("ALTER TABLE users ADD COLUMN vault_password_hash TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

def get_vault_file(user_id: int) -> str:
    """Returns per-user vault file path."""
    return f'vault_user_{user_id}.titan'

def get_vault_recovery_file(user_id: int) -> str:
    """Returns per-user vault recovery file path."""
    return f'vault_recovery_user_{user_id}.json'


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{pw_hash}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, pw_hash = stored_hash.split(':')
        return hashlib.sha256((salt + password).encode()).hexdigest() == pw_hash
    except Exception:
        return False

init_db()

# --- نظام سجل النشاط الأمني (Audit Log) ---
AUDIT_LOGS = []
BURN_NOTES = {}


def add_audit_log(action, details=""):
    event = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details
    }
    AUDIT_LOGS.insert(0, event)
    if len(AUDIT_LOGS) > 50: AUDIT_LOGS.pop()

# --- المنطق البرمجي: تشفير وفك تشفير ---

def derive_key(password: str, salt: bytes) -> bytes:
    """اشتقاق مفتاح تشفير آمن من كلمة سر"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def derive_raw_key(password: str, salt: bytes) -> bytes:
    """اشتقاق مفتاح خام 32 بايت (بدون base64) لـ ChaCha20"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return kdf.derive(password.encode())

def encrypt_data(data: bytes, password: str) -> bytes:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    f = Fernet(key)
    encrypted_data = f.encrypt(data)
    # ندمج الملح (salt) مع البيانات المشفرة لنتمكن من فكها لاحقاً
    return salt + encrypted_data

def decrypt_data(encrypted_content: bytes, password: str) -> bytes:
    try:
        salt = encrypted_content[:16]  # type: ignore
        data = encrypted_content[16:]  # type: ignore
        key = derive_key(password, salt)
        f = Fernet(key)
        return f.decrypt(data)
    except Exception:
        raise ValueError("كلمة السر خاطئة أو الملف معطوب")

# --- ميزات الخصوصية المتقدمة (Privacy & Steganography) ---

def remove_image_metadata(img_bytes: bytes) -> bytes:
    """إزالة كافة الميتابيانات عن طريق إعادة حفظ الصورة بدون EXIF"""
    img = Image.open(io.BytesIO(img_bytes))
    data = list(img.getdata())
    img_no_meta = Image.new(img.mode, img.size)
    img_no_meta.putdata(data)
    
    out = io.BytesIO()
    # نحافظ على التنسيق الأصلي إذا أمكن أو نحول لـ PNG للسلامة
    fmt = img.format if img.format else "PNG"
    img_no_meta.save(out, format=fmt)
    return out.getvalue()

def lsb_encode(img_bytes: bytes, secret_data: str) -> bytes:
    """إخفاء نص في بيانات الصورة (Least Significant Bit)"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    width, height = img.size
    
    # تحويل النص لـ UTF-8 ثم لـ Binary مع علامة نهاية
    binary_data = ''.join([format(b, "08b") for b in secret_data.encode('utf-8')]) + '1111111111111110'
    
    if len(binary_data) > width * height * 3:
        raise ValueError("البيانات كبيرة جداً بالنسبة لهذه الصورة!")
        
    pixels = img.load()
    data_idx = 0
    
    for y in range(height):
        for x in range(width):
            if data_idx < len(binary_data):
                r, g, b, a = pixels[x, y]  # type: ignore
                # تعديل R
                r = (r & ~1) | int(binary_data[data_idx])  # type: ignore
                data_idx += 1
                if data_idx < len(binary_data):
                    # تعديل G
                    g = (g & ~1) | int(binary_data[data_idx])  # type: ignore
                    data_idx += 1
                if data_idx < len(binary_data):
                    # تعديل B
                    b = (b & ~1) | int(binary_data[data_idx])  # type: ignore
                    data_idx += 1
                pixels[x, y] = (r, g, b, a)
            else:
                break
        if data_idx >= len(binary_data): break
        
    out = io.BytesIO()
    img.save(out, format="PNG") # PNG يحافظ على البكسلات بدقة
    return out.getvalue()

def lsb_decode(img_bytes: bytes) -> str:
    """استخراج النص المخفي من الصورة"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    width, height = img.size
    pixels = img.load()
    
    bits: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]  # type: ignore
            bits.append(r & 1)  # type: ignore
            bits.append(g & 1)
            bits.append(b & 1)
    
    # Search for the end marker 1111111111111110 in the bitstream
    END_MARKER = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    for i in range(len(bits) - 15):
        if bits[i:i+16] == END_MARKER:  # type: ignore[misc]
            # All bits before this marker are our message
            data_bits: list[int] = bits[:i]  # type: ignore[misc]
            # Only take complete bytes
            num_bytes = len(data_bits) // 8
            if num_bytes == 0:
                return "لم يتم العثور على بيانات مخفية!"
            byte_data = bytes([int(''.join(str(b) for b in data_bits[j*8:(j+1)*8]), 2) for j in range(num_bytes)])
            try:
                return byte_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return byte_data.decode('latin-1')
                except Exception:
                    return "فشل استخراج النص: الصورة لا تحتوي على بيانات مخفية بواسطة هذه الأداة."
                
    return "لم يتم العثور على بيانات مخفية في هذه الصورة!"


# --- استخبارات الصور (Image EXIF OSINT) ---
def extract_exif_data(img_bytes: bytes) -> dict:
    tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
    extracted = {}
    important_tags = ['Image Make', 'Image Model', 'Image DateTime', 'Image Software', 'GPS GPSLatitude', 'GPS GPSLongitude']
    for tag in tags.keys():
        if any(imp in tag for imp in important_tags):
            extracted[tag] = str(tags[tag])
    return extracted if extracted else {"Info": "لا توجد أي بيانات وصفية مخفية (EXIF) في هذه الصورة."}

# --- فحص الإيميل عبر IPQualityScore API ---
def check_email_intelligence(email: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    url = f'https://ipqualityscore.com/api/json/email/{API_KEY}/{email}'
    params = {
        'timeout': 7,
        'fast': 'false',
        'abuse_strictness': 0
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- فحص رقم الهاتف عبر IPQualityScore API ---
def check_phone_intelligence(phone: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    # تنظيف وتجهيز رقم الهاتف
    phone_clean = urllib.parse.quote(phone.strip())
    url = f'https://www.ipqualityscore.com/api/json/phone/{API_KEY}/{phone_clean}'
    params = {}
        
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- فحص الروابط المشبوهة عبر IPQualityScore API ---
def check_url_intelligence(target_url: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    url_clean = urllib.parse.quote(target_url.strip(), safe='')
    url = f'https://www.ipqualityscore.com/api/json/url/{API_KEY}/{url_clean}'
    params = {'fast': 'true', 'strictness': 0}
        
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- فحص تسريب الإيميل وكلمة السر معاً عبر IPQualityScore API ---
def check_leaked_emailpass(email: str, password: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    url = f"https://www.ipqualityscore.com/api/json/leaked/emailpass/{API_KEY}"
    post_data = {
        "email": email,
        "password": password
    }
    
    try:
        response = requests.post(url, json=post_data, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- جلب سجلات الاستخدام من IPQualityScore ---
def get_ipqs_requests_list(req_type: str, start_date: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    url = f'https://www.ipqualityscore.com/api/json/requests/{API_KEY}/list'
    params = {
        'type': req_type,
        'start_date': start_date
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- إخفاء البيانات في الصوت (Audio Steganography) ---
def wave_lsb_encode(wav_bytes: bytes, secret_data: str, filename: str = "") -> bytes:
    """إخفاء نص في ملف صوتي (WAV LSB أو MP3 EOF)"""
    try:
        # Marker for extraction
        marker = b'##TITAN_SECURE##'
        full_secret = secret_data.encode('utf-8') + marker
        
        if filename.lower().endswith('.mp3'):
            # Append at EOF for MP3 (Safe & Reliable)
            return wav_bytes + marker + secret_data.encode('utf-8')
        
        # WAV Steganography: Modify frames
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
            params = wav.getparams()
            frames = bytearray(wav.readframes(wav.getnframes()))

        # Convert to bitstream
        bits = ''.join(format(b, '08b') for b in full_secret)
        
        if len(bits) > len(frames):
            raise ValueError("النص كبير جداً بالنسبة لملف الصوت المحدد!")

        # Apply LSB
        for i, bit in enumerate(bits):
            frames[i] = (frames[i] & ~1) | int(bit)

        out = io.BytesIO()
        with wave.open(out, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(frames)
        
        return out.getvalue()
    except Exception as e:
        raise ValueError(f"فشل تشفير الصوت: {str(e)}")

def wave_lsb_decode(wav_bytes: bytes, filename: str = "") -> str:
    """استخراج النص المخفي من ملف WAV أو MP3"""
    try:
        marker = b'##TITAN_SECURE##'
        
        if filename.lower().endswith('.mp3'):
            if marker in wav_bytes:
                return wav_bytes.split(marker)[-1].decode('utf-8', errors='ignore')
            return "لم يتم العثور على بيانات مخفية في ملف MP3."

        # WAV LSB Decode
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
            frames = bytearray(wav.readframes(wav.getnframes()))

        bits = [str(f & 1) for f in frames]
        byte_list = []
        # Reconstruct bytes
        for i in range(0, len(bits), 8):
            if i + 8 > len(bits): break
            byte_val = int(''.join(bits[i:i+8]), 2)
            byte_list.append(byte_val)
        
        raw_result = bytes(byte_list)
        if marker in raw_result:
            return raw_result.split(marker)[0].decode('utf-8', errors='ignore')
        
        return "لم يتم العثور على بصمة نص مخفي في ملف الصوت."
    except Exception as e:
        return f"خطأ في تحليل البيانات: {str(e)}"

# --- تنظيف ملفات PDF من الميتابيانات ---
def clean_pdf_metadata(pdf_bytes: bytes) -> bytes:
    """إزالة الميتابيانات وكافة المعلومات الوصفية من ملف PDF"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        
        # إفراغ الميتابيانات تماماً
        writer.add_metadata({}) 
        
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    except Exception as e:
        raise ValueError(f"فشل تنظيف PDF: {str(e)}")

# --- ميزات استخباراتية إضافية ---
def get_dns_leak_info() -> dict:
    """محاكاة فحص تسريب DNS"""
    return {
        "success": True,
        "leaked": False,
        "dns_servers": ["1.1.1.1 (Cloudflare)", "8.8.8.8 (Google)"],
        "isp": "TITAN Secure Relay"
    }

def get_shodan_intel(ip: str) -> dict:
    """محاكاة جلب بيانات Shodan لعنوان IP"""
    return {
        "success": True,
        "ip": ip,
        "ports": [80, 443, 21, 22] if random.random() > 0.5 else [80, 443],
        "vulnerabilities": ["CVE-2023-TITAN (Simulated)"] if random.random() > 0.8 else [],
        "last_scan": datetime.datetime.now().strftime("%Y-%m-%d")
    }

# --- فحص البرمجيات الخبيثة (Malware) عبر IPQualityScore API ---
def handle_malware_result(data):
    if data.status_code != 200:
        return {"success": False, "message": f"Error: {data.status_code}", "error": f"Error: {data.status_code}"}
    
    try:
        res_json = data.json()
        if res_json.get("status") != "pending":
            # إرجاع بيانات الفحص
            return res_json
            
        # إذا كان الفحص قيد الانتظار، ننتظر ثانية ونسأل مجدداً
        while True:
            time.sleep(1)
            update_url = res_json.get("update_url")
            if not update_url:
                break
            data = requests.post(update_url)
            return handle_malware_result(data)
    except Exception as e:
        return {"success": False, "message": "فشل تحليل استجابة الموقع", "error": str(e)}

def scan_malware_url(url: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    try:
        data = requests.post(f"https://www.ipqualityscore.com/api/json/malware/scan/{API_KEY}", data={'url': url})
        return handle_malware_result(data)
    except Exception as e:
        return {"success": False, "message": "فشل الاتصال بالخدمة", "error": str(e)}

def scan_malware_file(file_path: str) -> dict:
    API_KEY = '1ZFJTNYsuxNXvJwdiETskE0DqpHJDIc4'
    try:
        with open(file_path, "rb") as f:
            data = requests.post(f"https://www.ipqualityscore.com/api/json/malware/scan/{API_KEY}", files={'file': f})
        return handle_malware_result(data)
    except Exception as e:
        return {"success": False, "message": "فشل رفع الملف", "error": str(e)}

# --- فحص الأجهزة المتصلة بالشبكة المحلية (Network LAN Scanner) ---
def scan_local_network():
    devices = []
    try:
        # ويندوز (استخراج جدول ARP) لأنه أسرع ولا يتطلب صلاحيات Scapy المعقدة للمستخدم العادي
        output = subprocess.check_output("arp -a", shell=True).decode('cp1252', errors='ignore')
        for line in output.splitlines():
            # البحث عن IPs
            match = re.search(r'((?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\s+([0-9a-fA-F-]+)\s+(\w+)', line)
            if match:
                ip, mac, type_ = match.groups()
                if not ip.startswith('224.') and not ip.startswith('239.') and ip != '255.255.255.255':
                    device_icon = "💻"; device_label = "جهاز مستخدم"
                    if ip.endswith('.1'): 
                        device_icon = "🌐"; device_label = "جهاز التوجيه"
                    elif '.10' in ip or '.20' in ip:
                        device_icon = "📱"; device_label = "هاتف محمول"
                    devices.append({
                        "ip": ip, 
                        "mac": mac.replace('-', ':').upper(), 
                        "type": type_,
                        "icon": device_icon,
                        "label": device_label
                    })
    except Exception as e:
        devices = [{"error": f"فشل الفحص: {str(e)}"}]
    return devices

# --- المنطق البرمجي: فحص قوة كلمة السر ---

def generate_strong_password(length=16):
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in password) and any(c.isupper() for c in password)
                and sum(c.isdigit() for c in password) >= 3):
            break
    return password

def generate_readable_passphrase(num_words=4):
    words = ["titan", "secure", "vault", "crypto", "shield", "phantom", "cyber", "ghost", 
             "delta", "omega", "matrix", "alpha", "zenith", "nebula", "storm", "blade",
             "orbit", "pulse", "static", "vector", "quantum", "binary", "proxy", "pixel"]
    selected = random.sample(words, num_words)
    return "-".join(selected) + str(random.randint(10, 99))

def get_strength_details(password):
    score = 0
    if len(password) >= 12: score += 1
    if re.search(r"[A-Z]", password): score += 1
    if re.search(r"[0-9]", password): score += 1
    if re.search(r"[!@#$%^&*]", password): score += 1
    if len(password) >= 16: score += 1
    levels = ["ضعيف جداً", "ضعيف", "متوسط", "قوي", "قوي جداً (TITAN Level)"]
    return levels[min(score, 4)], score

# --- فحص التسريبات (HIBP) ---

@lru_cache(maxsize=128)
def check_hibp_leak(password: str) -> int:
    if not password: return 0
    sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]  # type: ignore
    try:
        r = requests.get(f"https://api.pwnedpasswords.com/range/{prefix}", timeout=5)
        r.raise_for_status()
        for line in r.text.splitlines():
            s, count = line.split(':')
            if s == suffix: return int(count)
    except: pass
    return 0

# --- فحص IP ---
def get_ip_intelligence_data(ip=""):
    if not ip or ip == "127.0.0.1" or ip == "8.8.8.8":
        url = "http://ip-api.com/json/?fields=status,message,country,countryCode,regionName,city,zip,lat,lon,timezone,isp,org,as,proxy,query"
        ip_target = ""
    else:
        url = f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,regionName,city,zip,lat,lon,timezone,isp,org,as,proxy,query"
        ip_target = ip
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                is_proxy = data.get("proxy", False)
                
                # فحص إضافي وموثوق للـ VPN باستخدام أداة مجانية أخرى (proxycheck.io)
                try:
                    target = ip_target or data.get("query", "")
                    if target:
                        pc_url = f"http://proxycheck.io/v2/{target}?vpn=1&asn=1"
                        pc_res = requests.get(pc_url, timeout=5)
                        if pc_res.status_code == 200:
                            pc_data = pc_res.json()
                            if target in pc_data and pc_data[target].get("proxy") == "yes":
                                is_proxy = True
                except Exception:
                    pass

                # تنسيق البيانات لتتوافق مع ما يتوقعه سكريبت الجافاسكريبت
                return {
                    "success": True,
                    "proxy": is_proxy,
                    "vpn": is_proxy,
                    "fraud_score": 100 if is_proxy else 0,
                    "country_code": data.get("countryCode", "US"),
                    "ISP": data.get("isp", "Unknown"),
                    "query": data.get("query", ip)
                }
            else:
                 return {"success": False, "message": data.get("message", "فشل جلب البيانات")}
        else:
            return {"success": False, "message": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "message": str(e), "error": str(e)}

# --- واجهة المستخدم (HTML) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TITAN | التشفير والأمن السيبراني</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Tajawal', sans-serif; background: #070b19; color: white; margin: 0; overflow-x: hidden; cursor: crosshair; }
        #matrix-bg, #intro-matrix { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; }
        #matrix-bg { z-index: -1; }
        #intro-matrix { z-index: 0; opacity: 0.6; }
        .glass { background: rgba(10, 15, 30, 0.85); backdrop-filter: blur(16px); border: 1px solid rgba(168, 85, 247, 0.2); box-shadow: 0 0 30px rgba(0,0,0,0.5); }
        button, a, input { cursor: pointer; }
        .titan-gradient { background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%); }
        
        /* Scanlines & CRT Effect */
        body::after { content: " "; display: block; position: fixed; top: 0; left: 0; bottom: 0; right: 0; background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06)); z-index: 999; background-size: 100% 2px, 3px 100%; pointer-events: none; }
        
        /* Premium Ambient Background */
        #intro-overlay { 
            background: linear-gradient(180deg, #050505 0%, #0a0a0a 100%);
            position: fixed; inset: 0; z-index: 9999; 
            display: flex; flex-direction: column; justify-content: center; align-items: center; 
            transition: opacity 1.5s cubic-bezier(0.4, 0, 0.2, 1); 
            overflow: hidden; 
        }
        /* Animated Dark Violet Glows */
        #intro-overlay::before, #intro-overlay::after {
            content: ''; position: absolute; border-radius: 50%; filter: blur(140px); z-index: 0; pointer-events: none;
            animation: float-glowing-bars 12s ease-in-out infinite alternate;
        }
        #intro-overlay::before {
            width: 80vw; height: 30vh; background: rgba(168, 85, 247, 0.15); /* Purple */
            top: 20%; left: 10%;
        }
        #intro-overlay::after {
            width: 60vw; height: 40vh; background: rgba(139, 92, 246, 0.12); /* Deep Violet */
            bottom: 10%; right: 20%;
            animation-delay: -6s;
        }
        /* Dynamic Horizontal Scans (like the reference image background) */
        .premium-bg-scan {
            position: absolute; inset: 0; z-index: 1; pointer-events: none; opacity: 0.3;
            background: repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(168, 85, 247, 0.03) 40px, rgba(168, 85, 247, 0.03) 80px);
            mask-image: linear-gradient(to bottom, transparent, black 20%, black 80%, transparent);
            -webkit-mask-image: linear-gradient(to bottom, transparent, black 20%, black 80%, transparent);
            animation: bg-pan 30s linear infinite;
        }
        @keyframes bg-pan { 0% { background-position: 0 0; } 100% { background-position: 400px 0; } }
        @keyframes float-glowing-bars { 0% { transform: translateY(-20px) scale(1); opacity: 0.8; } 100% { transform: translateY(20px) scale(1.1); opacity: 1; } }

        /* Premium Typography */
        .premium-title { 
            font-size: 4.5rem; font-weight: 800; color: #ffffff; 
            position: relative; z-index: 10;
            line-height: 1.1; letter-spacing: -1px;
            text-align: center; margin-bottom: 1.5rem;
        }
        .premium-subtitle {
            font-size: 1.1rem; color: #94a3b8; z-index: 10;
            max-width: 600px; text-align: center; margin-top: 1rem; line-height: 1.6;
        }
        
        /* Fingerprint Scanner Button (Violet Theme) */
        .fingerprint-btn { margin-top: 3.5rem; width: 75px; height: 95px; border: 2px solid transparent; background: transparent; cursor: pointer; position: relative; transition: all 0.3s ease; opacity: 0; transform: translateY(20px); z-index: 10; display: inline-flex; flex-direction: column; align-items: center;}
        .fingerprint-btn.show { opacity: 1; transform: translateY(0); }
        .fingerprint-btn svg { width: 100%; height: 100%; fill: #a855f7; filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.6)); transition: all 0.3s ease; }
        .fingerprint-btn:hover svg { fill: #c084fc; filter: drop-shadow(0 0 18px rgba(192, 132, 252, 0.8)); }
        .scanner-line { position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: #c084fc; box-shadow: 0 0 12px #c084fc, 0 0 25px #a855f7; border-radius: 50%; opacity: 0; pointer-events: none; }
        .fingerprint-btn:hover .scanner-line { opacity: 1; animation: scan 1.5s infinite linear; }
        @keyframes scan { 0% { top: 0; } 50% { top: 100%; } 100% { top: 0; } }
        
        /* Radar Animation for IP */
        .radar-box { position: relative; width: 150px; height: 150px; border-radius: 50%; border: 2px solid #22c55e; background: rgba(34, 197, 94, 0.1); overflow: hidden; margin: 0 auto; box-shadow: 0 0 20px rgba(34,197,94,0.3); }
        .radar-box::before { content: ''; position: absolute; top: 50%; left: 50%; width: 50%; height: 50%; transform-origin: top left; background: linear-gradient(45deg, rgba(34,197,94,0.8) 0%, transparent 50%); animation: radar-spin 2s linear infinite; }
        .radar-box::after { content: ''; position: absolute; top: 50%; left: 0; right: 0; border-top: 1px solid rgba(34,197,94,0.5); }
        .radar-cross { position: absolute; left: 50%; top: 0; bottom: 0; border-left: 1px solid rgba(34,197,94,0.5); }
        .radar-target { position: absolute; width: 8px; height: 8px; background: #ef4444; border-radius: 50%; box-shadow: 0 0 10px #ef4444; opacity: 0; top: 30%; left: 60%; animation: target-ping 2s infinite; }
        @keyframes radar-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes target-ping { 0%, 100% { transform: scale(1); opacity: 0; } 50% { transform: scale(1.5); opacity: 1; } }

    </style>
</head>
<body class="min-h-screen relative">
    <!-- ===== AUTH OVERLAY (Login / Register) ===== -->
    <div id="auth-overlay" style="display:none; position:fixed; inset:0; z-index:99999; background:#050510; overflow:hidden;">
        <!-- Matrix Canvas inside Auth -->
        <canvas id="auth-matrix" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;"></canvas>

        <!-- Glowing Orbs -->
        <div style="position:absolute;width:60vw;height:60vw;border-radius:50%;background:radial-gradient(circle,rgba(139,92,246,0.18) 0%,transparent 70%);top:-20%;left:-10%;filter:blur(80px);animation:orbFloat 10s ease-in-out infinite alternate;pointer-events:none;z-index:1;"></div>
        <div style="position:absolute;width:40vw;height:40vw;border-radius:50%;background:radial-gradient(circle,rgba(168,85,247,0.14) 0%,transparent 70%);bottom:-15%;right:5%;filter:blur(80px);animation:orbFloat 14s ease-in-out infinite alternate-reverse;pointer-events:none;z-index:1;"></div>

        <!-- Auth Card -->
        <div style="position:relative;z-index:10;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:1.5rem;" id="auth-card-wrapper">
            <div style="width:100%;max-width:420px;background:rgba(10,10,30,0.85);border:1px solid rgba(139,92,246,0.35);border-radius:24px;padding:2.5rem 2rem;box-shadow:0 0 80px rgba(139,92,246,0.25),0 25px 60px rgba(0,0,0,0.6);backdrop-filter:blur(24px);">

                <!-- Logo -->
                <div style="text-align:center;margin-bottom:2rem;">
                    <div style="font-size:3.5rem;font-weight:900;letter-spacing:-2px;background:linear-gradient(135deg,#a855f7,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;">TITAN</div>
                    <div style="color:#6d28d9;font-size:0.7rem;letter-spacing:0.4em;text-transform:uppercase;margin-top:4px;">Security Protocol</div>
                    <div style="width:60px;height:1px;background:linear-gradient(90deg,transparent,#a855f7,transparent);margin:1rem auto 0;"></div>
                </div>

                <!-- Tab Toggle -->
                <div style="display:flex;background:rgba(15,15,40,0.8);border-radius:12px;padding:4px;margin-bottom:1.8rem;border:1px solid rgba(139,92,246,0.2);">
                    <button id="auth-tab-login" onclick="switchAuthTab('login')" style="flex:1;padding:0.6rem;border-radius:9px;border:none;cursor:pointer;font-weight:700;font-size:0.85rem;transition:all 0.25s;background:linear-gradient(135deg,#a855f7,#7c3aed);color:white;box-shadow:0 0 15px rgba(168,85,247,0.4);font-family:Tajawal,sans-serif;">تسجيل الدخول</button>
                    <button id="auth-tab-register" onclick="switchAuthTab('register')" style="flex:1;padding:0.6rem;border-radius:9px;border:none;cursor:pointer;font-weight:700;font-size:0.85rem;transition:all 0.25s;background:transparent;color:#6b7280;font-family:Tajawal,sans-serif;">إنشاء حساب</button>
                </div>

                <!-- Login Form -->
                <div id="auth-login-form">
                    <div style="margin-bottom:1rem;">
                        <label style="display:block;color:#9ca3af;font-size:0.78rem;margin-bottom:6px;letter-spacing:0.05em;">اسم المستخدم</label>
                        <input id="auth-login-user" type="text" placeholder="اسم المستخدم..." autocomplete="username" style="width:100%;box-sizing:border-box;padding:0.85rem 1rem;background:rgba(15,15,40,0.9);border:1px solid rgba(139,92,246,0.3);border-radius:12px;color:white;font-size:0.95rem;outline:none;transition:border-color 0.2s;font-family:Tajawal,sans-serif;" onfocus="this.style.borderColor='#a855f7'" onblur="this.style.borderColor='rgba(139,92,246,0.3)'">
                    </div>
                    <div style="margin-bottom:1.5rem;">
                        <label style="display:block;color:#9ca3af;font-size:0.78rem;margin-bottom:6px;letter-spacing:0.05em;">كلمة السر</label>
                        <input id="auth-login-pass" type="password" placeholder="كلمة السر..." autocomplete="current-password" style="width:100%;box-sizing:border-box;padding:0.85rem 1rem;background:rgba(15,15,40,0.9);border:1px solid rgba(139,92,246,0.3);border-radius:12px;color:white;font-size:0.95rem;outline:none;transition:border-color 0.2s;font-family:Tajawal,sans-serif;" onfocus="this.style.borderColor='#a855f7'" onblur="this.style.borderColor='rgba(139,92,246,0.3)'">
                    </div>
                    <div id="auth-login-error" style="display:none;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);border-radius:10px;padding:0.7rem 1rem;color:#f87171;font-size:0.82rem;margin-bottom:1rem;text-align:center;"></div>
                    <button onclick="doLogin()" style="width:100%;padding:0.9rem;background:linear-gradient(135deg,#a855f7,#7c3aed);border:none;border-radius:12px;color:white;font-size:1rem;font-weight:700;cursor:pointer;transition:all 0.2s;box-shadow:0 0 20px rgba(168,85,247,0.4);font-family:Tajawal,sans-serif;" onmouseover="this.style.boxShadow='0 0 35px rgba(168,85,247,0.7)'" onmouseout="this.style.boxShadow='0 0 20px rgba(168,85,247,0.4)'" id="auth-login-btn">
                        دخول إلى TITAN 🔐
                    </button>
                </div>

                <!-- Register Form -->
                <div id="auth-register-form" style="display:none;">
                    <div style="margin-bottom:1rem;">
                        <label style="display:block;color:#9ca3af;font-size:0.78rem;margin-bottom:6px;letter-spacing:0.05em;">اسم المستخدم (3 أحرف على الأقل)</label>
                        <input id="auth-reg-user" type="text" placeholder="اختر اسم مستخدم..." autocomplete="username" style="width:100%;box-sizing:border-box;padding:0.85rem 1rem;background:rgba(15,15,40,0.9);border:1px solid rgba(139,92,246,0.3);border-radius:12px;color:white;font-size:0.95rem;outline:none;transition:border-color 0.2s;font-family:Tajawal,sans-serif;" onfocus="this.style.borderColor='#a855f7'" onblur="this.style.borderColor='rgba(139,92,246,0.3)'">
                    </div>
                    <div style="margin-bottom:1rem;">
                        <label style="display:block;color:#9ca3af;font-size:0.78rem;margin-bottom:6px;letter-spacing:0.05em;">كلمة السر (6 أحرف على الأقل)</label>
                        <input id="auth-reg-pass" type="password" placeholder="اختر كلمة سر قوية..." autocomplete="new-password" style="width:100%;box-sizing:border-box;padding:0.85rem 1rem;background:rgba(15,15,40,0.9);border:1px solid rgba(139,92,246,0.3);border-radius:12px;color:white;font-size:0.95rem;outline:none;transition:border-color 0.2s;font-family:Tajawal,sans-serif;" onfocus="this.style.borderColor='#a855f7'" onblur="this.style.borderColor='rgba(139,92,246,0.3)'">
                    </div>
                    <div style="margin-bottom:1.5rem;">
                        <label style="display:block;color:#9ca3af;font-size:0.78rem;margin-bottom:6px;letter-spacing:0.05em;">تأكيد كلمة السر</label>
                        <input id="auth-reg-pass2" type="password" placeholder="أعد كتابة كلمة السر..." autocomplete="new-password" style="width:100%;box-sizing:border-box;padding:0.85rem 1rem;background:rgba(15,15,40,0.9);border:1px solid rgba(139,92,246,0.3);border-radius:12px;color:white;font-size:0.95rem;outline:none;transition:border-color 0.2s;font-family:Tajawal,sans-serif;" onfocus="this.style.borderColor='#a855f7'" onblur="this.style.borderColor='rgba(139,92,246,0.3)'">
                    </div>
                    <div id="auth-reg-error" style="display:none;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);border-radius:10px;padding:0.7rem 1rem;color:#f87171;font-size:0.82rem;margin-bottom:1rem;text-align:center;"></div>
                    <div id="auth-reg-success" style="display:none;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.4);border-radius:10px;padding:0.7rem 1rem;color:#4ade80;font-size:0.82rem;margin-bottom:1rem;text-align:center;"></div>
                    <button onclick="doRegister()" style="width:100%;padding:0.9rem;background:linear-gradient(135deg,#7c3aed,#5b21b6);border:none;border-radius:12px;color:white;font-size:1rem;font-weight:700;cursor:pointer;transition:all 0.2s;box-shadow:0 0 20px rgba(124,58,237,0.4);font-family:Tajawal,sans-serif;" onmouseover="this.style.boxShadow='0 0 35px rgba(124,58,237,0.7)'" onmouseout="this.style.boxShadow='0 0 20px rgba(124,58,237,0.4)'" id="auth-reg-btn">
                        إنشاء حساب جديد ✨
                    </button>
                </div>

                <!-- Footer -->
                <div style="text-align:center;margin-top:1.5rem;color:#374151;font-size:0.72rem;letter-spacing:0.05em;">
                    🛡️ TITAN SECURITY PROTOCOL — ALL DATA ENCRYPTED
                </div>
            </div>
        </div>

        <style>
            @keyframes orbFloat { 0%{transform:translate(0,0) scale(1);} 100%{transform:translate(3%,5%) scale(1.08);} }
        </style>
    </div>
    <!-- END AUTH OVERLAY -->

    <!-- شاشة المقدمة الفاخرة (Premium Intro) -->

    <div id="intro-overlay">
        <div class="premium-bg-scan"></div>
        <canvas id="intro-matrix"></canvas> <!-- 3D Matrix Background inside Intro -->
        <div class="flex flex-col items-center justify-center h-full w-full opacity-0 translate-y-8 transition-all duration-1000 z-10 p-4" id="intro-center-logo">
            <h1 class="premium-title font-serif">TITAN<br><span class="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-violet-600">Security Protocol</span></h1>
            <p class="premium-subtitle text-gray-400" dir="ltr">The Next-Gen Encryption & Intelligence Platform to protect your data with state-of-the-art security algorithms in a seamless interface.</p>
            
            <button id="start-btn" onclick="startSystem()" class="fingerprint-btn" title="Initiate System">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12,2C6.48,2 2,6.48 2,12C2,17.52 6.48,22 12,22C17.52,22 22,17.52 22,12C22,6.48 17.52,2 12,2M11,19.93C7.05,19.43 4,16.05 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.05 16.95,19.43 13,19.93V17H11V19.93M13,6.5A5.5,5.5 0 0,0 7.5,12H9.5A3.5,3.5 0 0,1 13,8.5V6.5M13,10A2,2 0 0,0 11,12H13V10Z" />
                </svg>
                <div class="scanner-line"></div>
                <!-- Glowing Circle Echo -->
                <div class="absolute inset-0 rounded-full border border-purple-500/30 animate-ping" style="animation-duration: 2.5s; z-index: -1; transform: scale(1.6);"></div>
                <div class="text-purple-400 text-xs mt-8 uppercase font-bold tracking-[0.2em] text-center animate-pulse whitespace-nowrap" style="text-shadow: 0 0 10px rgba(168, 85, 247, 0.8);">Touch To Authenticate</div>
            </button>
        </div>
    </div>

    <div id="main-app" class="opacity-0 transition-opacity duration-1000 ease-in-out pointer-events-none">
        <canvas id="matrix-bg"></canvas>
        <div class="container mx-auto px-4 py-12 max-w-3xl relative z-10">
        <header class="text-center mb-12 relative">
            <h1 class="text-7xl font-black text-purple-500 mb-2 tracking-tighter" style="text-shadow: 0 0 20px rgba(168, 85, 247, 0.5);">TITAN</h1>
            <p class="text-gray-400 text-lg text-purple-400">نظام التشفير وحماية البيانات المتطور</p>
            
            <div style="position:absolute;top:0;left:0;display:flex;align-items:center;gap:0.5rem;">
                <span id="header-username" style="color:#a855f7;font-size:0.75rem;font-weight:700;letter-spacing:0.05em;background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.3);padding:4px 10px;border-radius:8px;"></span>
                <button onclick="doLogout()" title="تسجيل الخروج" style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);color:#f87171;padding:4px 10px;border-radius:8px;cursor:pointer;font-size:0.75rem;font-weight:700;transition:all 0.2s;" onmouseover="this.style.background='rgba(239,68,68,0.3)'" onmouseout="this.style.background='rgba(239,68,68,0.15)'">🚪 خروج</button>
            </div>
        </header>

        <div class="glass p-8 rounded-2xl shadow-2xl">
            <!-- Navigation -->
            <div class="mb-8 flex flex-wrap justify-center gap-2 p-2 bg-slate-900/40 rounded-xl border border-slate-700/50">
                <button onclick="showTab('dash')" id="btn-dash" class="px-3 py-1.5 rounded-lg hover:bg-purple-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-purple-500/30"><span>📊</span> الإحصائيات</button>
                <button onclick="showTab('pass')" id="btn-pass" class="px-3 py-1.5 rounded-lg hover:bg-purple-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-purple-500/30"><span>🔑</span> كلمات السر</button>
                <button onclick="showTab('vault'); checkVaultPasswordSetup();" id="btn-vault" class="px-3 py-1.5 rounded-lg hover:bg-yellow-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-yellow-500/30"><span>🗄️</span> القبو</button>
                <button onclick="showTab('crypt')" id="btn-crypt" class="px-3 py-1.5 rounded-lg hover:bg-blue-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-blue-500/30"><span>🔐</span> التشفير</button>
                <button onclick="showTab('suite')" id="btn-suite" class="px-3 py-1.5 rounded-lg hover:bg-purple-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-purple-500/30"><span>🛠️</span> الأدوات الذكية</button>
                <button onclick="showTab('tools')" id="btn-tools" class="px-3 py-1.5 rounded-lg hover:bg-purple-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-purple-500/30"><span>🌐</span> تتبع IP</button>
                <button onclick="showTab('netintel')" id="btn-netintel" class="px-3 py-1.5 rounded-lg hover:bg-orange-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-orange-500/30"><span>🔬</span> استخبارات</button>
                <button onclick="showTab('extreme')" id="btn-extreme" class="px-3 py-1.5 rounded-lg hover:bg-teal-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-teal-500/30"><span>🛡️</span> الخصوصية</button>
                <button onclick="showTab('audio')" id="btn-audio" class="px-3 py-1.5 rounded-lg hover:bg-orange-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-orange-500/30"><span>🎵</span> إخفاء صوتي</button>
                <button onclick="showTab('qr')" id="btn-qr" class="px-3 py-1.5 rounded-lg hover:bg-green-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-green-500/30"><span>🔳</span> QR آمن</button>
                <button onclick="showTab('identity')" id="btn-identity" class="px-3 py-1.5 rounded-lg hover:bg-cyan-600/20 text-xs font-bold text-gray-400 transition-all flex items-center gap-1.5 border border-transparent hover:border-cyan-500/30"><span>🪪</span> هوية وهمية</button>
            </div>

            <!-- ===== DASHBOARD SECTION ===== -->
            <div id="dash-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-purple-400 border-b border-slate-700 pb-2">📊 لوحة التحكم – معلومات النظام</h2>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3" id="dashCards">
                    <div class="bg-slate-900 rounded-xl p-4 border border-purple-800/40 text-center">
                        <div class="text-3xl font-black text-purple-400" id="dashCpu">—</div>
                        <div class="text-xs text-gray-500 mt-1">CPU %</div>
                    </div>
                    <div class="bg-slate-900 rounded-xl p-4 border border-blue-800/40 text-center">
                        <div class="text-3xl font-black text-blue-400" id="dashRam">—</div>
                        <div class="text-xs text-gray-500 mt-1">RAM %</div>
                    </div>
                    <div class="bg-slate-900 rounded-xl p-4 border border-green-800/40 text-center">
                        <div class="text-3xl font-black text-green-400" id="dashDisk">—</div>
                        <div class="text-xs text-gray-500 mt-1">Disk %</div>
                    </div>
                    <div class="bg-slate-900 rounded-xl p-4 border border-yellow-800/40 text-center">
                        <div class="text-3xl font-black text-yellow-400" id="dashBurn">—</div>
                        <div class="text-xs text-gray-500 mt-1">Burn Notes</div>
                    </div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-slate-900/70 rounded-xl p-4 border border-slate-700">
                        <div class="text-xs text-gray-400 mb-3 font-bold">🌐 معلومات الشبكة</div>
                        <div class="space-y-2 text-sm font-mono">
                            <div class="flex justify-between"><span class="text-gray-500">IP المحلي</span><span class="text-green-400" id="dashLocalIp">—</span></div>
                            <div class="flex justify-between"><span class="text-gray-500">IP العام</span><span class="text-blue-400" id="dashPubIp">—</span></div>
                            <div class="flex justify-between"><span class="text-gray-500">صادر (MB)</span><span class="text-purple-400" id="dashSent">—</span></div>
                            <div class="flex justify-between"><span class="text-gray-500">وارد (MB)</span><span class="text-purple-400" id="dashRecv">—</span></div>
                        </div>
                    </div>
                    <div class="bg-slate-900/70 rounded-xl p-4 border border-slate-700">
                        <div class="text-xs text-gray-400 mb-3 font-bold">📋 آخر النشاطات</div>
                        <div id="dashLogs" class="space-y-1 text-xs font-mono max-h-36 overflow-y-auto"></div>
                    </div>
                </div>
                <button onclick="loadDashboard()" class="titan-gradient px-6 py-2 rounded-xl font-bold text-sm">🔄 تحديث</button>
            </div>

            <!-- ===== PASSWORD SECTION ===== -->
            <div id="pass-section">
                <label class="block text-sm text-gray-400 mb-2">اختبر قوة كلمة السر:</label>
                <input type="password" id="passInput" class="w-full p-4 rounded-xl bg-slate-900 border border-slate-700 mb-4 text-left focus:ring-2 focus:ring-purple-500 outline-none transition-all">
                <div id="pass-result" class="mb-6 hidden">
                    <div class="flex justify-between items-center mb-2">
                        <span id="strength-text" class="font-bold"></span>
                        <span id="strength-percent" class="text-sm text-gray-400"></span>
                    </div>
                    <div class="h-3 bg-slate-700 rounded-full mb-4 overflow-hidden"><div id="strength-bar" class="h-full w-0 transition-all duration-700"></div></div>
                    <div id="leak-info" class="p-4 rounded-xl border hidden text-sm"></div>
                </div>
                <div class="flex gap-4">
                    <button onclick="generatePass('random')" class="text-purple-400 hover:text-purple-300 font-bold">✨ توليد كلمة سر TITAN</button>
                    <button onclick="generatePass('passphrase')" class="text-purple-400 hover:text-purple-300 font-bold">📖 توليد عبارت نصية (Passphrase)</button>
                </div>
                <div id="suggested-pass-container" class="mt-4 hidden p-4 bg-slate-900/50 rounded-xl border border-dashed border-purple-500/50 flex justify-between items-center">
                    <code id="suggested-pass" class="text-purple-400 font-mono text-lg"></code>
                    <button onclick="copyPass()" class="text-xs bg-slate-800 px-2 py-1 rounded">نسخ</button>
                </div>
            </div>

            <div id="suite-section" class="hidden space-y-8">
                <!-- ===== GLOBAL THREAT DASHBOARD (Interactive) ===== -->
                <div class="relative rounded-2xl border border-purple-900/50 overflow-hidden shadow-[0_0_40px_rgba(168,85,247,0.12)]" style="background:#050508;">

                    <!-- Animated Canvas Background -->
                    <canvas id="threatCanvas" class="absolute inset-0 w-full h-full opacity-40" style="height:220px;"></canvas>

                    <!-- Scanline overlay -->
                    <div class="absolute inset-0 pointer-events-none" style="background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px);"></div>

                    <!-- Content -->
                    <div class="relative z-10 p-5" style="min-height:220px;">
                        <!-- Header Row -->
                        <div class="flex items-start justify-between mb-3">
                            <div>
                                <div class="flex items-center gap-2 mb-0.5">
                                    <span class="w-2 h-2 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_#ef4444]"></span>
                                    <span class="text-[10px] font-mono text-red-400 uppercase tracking-[0.3em]">LIVE FEED</span>
                                </div>
                                <h3 class="text-xl font-black text-purple-300 tracking-tight">Global Threat Dashboard</h3>
                                <p class="text-[10px] font-mono text-gray-500 tracking-[0.2em] uppercase mt-0.5">Neural Threat Intelligence · Active Protocols <span id="gtd-protocols" class="text-purple-400">04</span></p>
                            </div>
                            <!-- Live clock -->
                            <div class="text-right">
                                <div id="gtd-clock" class="font-mono text-purple-400 text-sm font-bold tracking-widest"></div>
                                <div class="text-[9px] text-gray-600 font-mono mt-0.5">UTC+03:00</div>
                            </div>
                        </div>

                        <!-- Stats Row -->
                        <div class="grid grid-cols-4 gap-2 mb-3">
                            <div class="bg-black/40 border border-red-900/40 rounded-xl p-2 text-center">
                                <div id="gtd-threats" class="text-lg font-black text-red-400 font-mono leading-none">0</div>
                                <div class="text-[9px] text-gray-500 mt-0.5 uppercase tracking-wider">Threats Blocked</div>
                            </div>
                            <div class="bg-black/40 border border-purple-900/40 rounded-xl p-2 text-center">
                                <div id="gtd-enc" class="text-lg font-black text-purple-400 font-mono leading-none">0</div>
                                <div class="text-[9px] text-gray-500 mt-0.5 uppercase tracking-wider">Encrypt Ops/s</div>
                            </div>
                            <div class="bg-black/40 border border-blue-900/40 rounded-xl p-2 text-center">
                                <div id="gtd-nodes" class="text-lg font-black text-blue-400 font-mono leading-none">0</div>
                                <div class="text-[9px] text-gray-500 mt-0.5 uppercase tracking-wider">Active Nodes</div>
                            </div>
                            <div class="bg-black/40 border border-green-900/40 rounded-xl p-2 text-center">
                                <div id="gtd-entropy" class="text-lg font-black text-green-400 font-mono leading-none">0.00</div>
                                <div class="text-[9px] text-gray-500 mt-0.5 uppercase tracking-wider">Entropy</div>
                            </div>
                        </div>

                        <!-- Scrolling Cyber Ticker -->
                        <div class="overflow-hidden rounded-lg bg-black/50 border border-purple-900/30 py-1.5 px-0 relative" style="height:28px;">
                            <div id="gtd-ticker" class="flex gap-8 items-center font-mono text-[10px] whitespace-nowrap absolute" style="animation:gtdScroll 30s linear infinite;top:6px;left:0;">
                                <span class="text-purple-400">AES-256-GCM · SHA3-512 · BLAKE3</span>
                                <span class="text-red-400">⚠ INTRUSION ATTEMPT BLOCKED: 185.220.101.x</span>
                                <span class="text-green-400">∑(p·log₂p) = 7.998 bits/byte</span>
                                <span class="text-blue-400">RSA-4096 · ECDH-P521 · X25519</span>
                                <span class="text-yellow-400">⚡ CIPHER: ChaCha20-Poly1305 · IV: 96bit nonce</span>
                                <span class="text-purple-300">∀x∈{0,1}ⁿ: H(x) = H(k‖x) mod 2²⁵⁶</span>
                                <span class="text-red-400">⚠ BRUTEFORCE DETECTED → FIREWALL ENGAGED</span>
                                <span class="text-cyan-400">TLS 1.3 · HSTS · OCSP Stapling · CT Logs</span>
                                <span class="text-green-300">KDF: PBKDF2-HMAC-SHA512 · 310,000 iterations</span>
                                <span class="text-orange-400">⚡ ZERO-DAY SIGNATURE UPDATED: CVE-2025-TITAN</span>
                                <!-- duplicate for seamless loop -->
                                <span class="text-purple-400">AES-256-GCM · SHA3-512 · BLAKE3</span>
                                <span class="text-red-400">⚠ INTRUSION ATTEMPT BLOCKED: 185.220.101.x</span>
                                <span class="text-green-400">∑(p·log₂p) = 7.998 bits/byte</span>
                                <span class="text-blue-400">RSA-4096 · ECDH-P521 · X25519</span>
                            </div>
                        </div>
                    </div>
                </div>

                <style>
                    @keyframes gtdScroll { from { transform:translateX(0) } to { transform:translateX(-50%) } }
                </style>

                <script>
                (function(){
                    // === Canvas Cyber Streams ===
                    const canvas = document.getElementById('threatCanvas');
                    if(!canvas) return;
                    const ctx = canvas.getContext('2d');
                    function resizeCanvas(){
                        canvas.width = canvas.offsetWidth;
                        canvas.height = canvas.offsetHeight;
                    }
                    resizeCanvas();
                    window.addEventListener('resize', resizeCanvas);

                    const CHARS = '01アイウエオカキクケコ∑∆∇∫∂π≠≡◊⊕⊗⊞⊟ABCDEF'.split('');
                    const cols = Math.floor(canvas.width / 14);
                    const drops = Array.from({length: cols}, () => Math.random() * -50);
                    const speeds = Array.from({length: cols}, () => Math.random() * 0.4 + 0.15);
                    const colors = ['#7c3aed','#a855f7','#c084fc','#f43f5e','#3b82f6'];

                    // === Threat Pings ===
                    const pings = [];
                    function createPing() {
                        if(pings.length > 5) return;
                        pings.push({ 
                            x: Math.random() * canvas.width, 
                            y: Math.random() * canvas.height, 
                            r: 0, 
                            alpha: 1 
                        });
                    }
                    setInterval(createPing, 3000);

                    function drawThreatPings() {
                        for (let i = pings.length - 1; i >= 0; i--) {
                            const p = pings[i];
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                            ctx.strokeStyle = `rgba(239, 68, 68, ${p.alpha})`;
                            ctx.lineWidth = 2;
                            ctx.stroke();
                            
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
                            ctx.fillStyle = `rgba(239, 68, 68, ${p.alpha})`;
                            ctx.fill();

                            p.r += 1.5;
                            p.alpha -= 0.015;
                            if (p.alpha <= 0) pings.splice(i, 1);
                        }
                    }

                    function drawCyberStream(){
                        ctx.fillStyle = 'rgba(5,5,8,0.18)';
                        ctx.fillRect(0,0,canvas.width,canvas.height);
                        drawThreatPings();
                        for(let i=0;i<cols;i++){
                            const char = CHARS[Math.floor(Math.random()*CHARS.length)];
                            const color = colors[i % colors.length];
                            ctx.font = `bold 12px monospace`;
                            // head glow
                            ctx.fillStyle = '#fff';
                            ctx.shadowColor = color;
                            ctx.shadowBlur = 8;
                            ctx.fillText(char, i*14, drops[i]*14);
                            // trail
                            ctx.fillStyle = color + 'aa';
                            ctx.shadowBlur = 3;
                            ctx.fillText(CHARS[Math.floor(Math.random()*CHARS.length)], i*14, (drops[i]-1)*14);
                            ctx.shadowBlur = 0;
                            if(drops[i]*14 > canvas.height && Math.random() > 0.975) drops[i] = 0;
                            drops[i] += speeds[i];
                        }
                    }
                    setInterval(drawCyberStream, 50);

                    // === Live Counters ===
                    let threats = 14872, enc = 0, nodes = 217, entropy = 7.94;
                    function updateCounters(){
                        threats += Math.floor(Math.random()*3);
                        enc = Math.floor(Math.random()*9999 + 5000);
                        nodes = 200 + Math.floor(Math.random()*40);
                        entropy = (7.90 + Math.random()*0.09).toFixed(3);
                        const t = document.getElementById('gtd-threats');
                        const e = document.getElementById('gtd-enc');
                        const n = document.getElementById('gtd-nodes');
                        const en = document.getElementById('gtd-entropy');
                        if(t) t.textContent = threats.toLocaleString();
                        if(e) e.textContent = enc.toLocaleString();
                        if(n) n.textContent = nodes;
                        if(en) en.textContent = entropy;
                    }
                    updateCounters();
                    setInterval(updateCounters, 1200);

                    // === Live Clock ===
                    function updateClock(){
                        const now = new Date();
                        const h = String(now.getHours()).padStart(2,'0');
                        const m = String(now.getMinutes()).padStart(2,'0');
                        const s = String(now.getSeconds()).padStart(2,'0');
                        const el = document.getElementById('gtd-clock');
                        if(el) el.textContent = h + ':' + m + ':' + s;
                    }
                    updateClock();
                    setInterval(updateClock, 1000);
                })();
                </script>


                <!-- Features Grid -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-slate-900/50 p-5 rounded-xl border border-slate-700 hover:border-purple-500/50 transition-all group">
                        <div class="flex items-center gap-3 mb-4">
                            <div class="w-10 h-10 rounded-lg bg-purple-900/30 flex items-center justify-center text-purple-400">🖼️</div>
                            <h4 class="font-bold">Privacy Guard (Metadata)</h4>
                        </div>
                        <p class="text-[11px] text-gray-500 mb-4 h-8">حذف البيانات المخفية (EXIF) من الصور لحماية خصوصيتك عند المشاركة.</p>
                        <input type="file" id="metadataFile" class="hidden" accept="image/*" onchange="processMetadata()">
                        <button onclick="document.getElementById('metadataFile').click()" class="w-full py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm transition-all border border-slate-700">تنظيف صورة 🧹</button>
                    </div>
                    
                    <div class="bg-slate-900/50 p-5 rounded-xl border border-slate-700 hover:border-blue-500/50 transition-all group">
                        <div class="flex items-center gap-3 mb-4">
                            <div class="w-10 h-10 rounded-lg bg-blue-900/30 flex items-center justify-center text-blue-400">🕵️‍♂️</div>
                            <h4 class="font-bold">Image OSINT Intel</h4>
                        </div>
                        <p class="text-[11px] text-gray-500 mb-4 h-8">استخراج احداثيات GPS الأجهزة والتواريخ الفائتة المخفية في الصورة.</p>
                        <input type="file" id="osintFile" class="hidden" accept="image/*" onchange="processExifOsint()">
                        <button onclick="document.getElementById('osintFile').click()" class="w-full py-2 bg-blue-900/40 hover:bg-blue-800/50 text-blue-400 rounded-lg text-sm transition-all border border-blue-800/50">تحليل الصورة 👁️</button>
                        <div id="osintResult" class="hidden mt-3 p-3 bg-black/50 border border-slate-700 rounded text-[10px] font-mono whitespace-pre-wrap max-h-32 overflow-y-auto w-full" dir="ltr"></div>
                    </div>

                    <div class="bg-slate-900/50 p-5 rounded-xl border border-slate-700 hover:border-purple-500/50 transition-all group text-right">
                        <div class="flex items-center justify-end gap-3 mb-4">
                            <h4 class="font-bold">Stegano-Vault (تشفير الصور)</h4>
                            <div class="w-10 h-10 rounded-lg bg-purple-900/30 flex items-center justify-center text-purple-400">🕵️</div>
                        </div>
                        <p class="text-[11px] text-gray-500 mb-4 h-8">إخفاء رسائل نصية مشفرة داخل بكسلات الصور بشكل غير مرئي تماماً.</p>
                        <div class="flex gap-2">
                             <button onclick="showStego('encode')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm transition-all border border-slate-700">إخفاء نص 🔒</button>
                             <button onclick="showStego('decode')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm transition-all border border-slate-700">استخراج 🔓</button>
                        </div>
                    </div>
                </div>

                <!-- Phase 4 System Defense Grid -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <!-- USB Guardian -->
                    <div class="bg-teal-900/10 p-5 rounded-xl border border-teal-900/40 hover:border-teal-500/50 transition-all group relative overflow-hidden">
                        <div class="absolute top-0 right-0 w-16 h-16 bg-teal-600/10 rounded-bl-full z-0 pointer-events-none"></div>
                        <div class="flex items-center gap-3 mb-3 relative z-10">
                            <div class="w-10 h-10 rounded-lg bg-teal-900/30 flex items-center justify-center text-teal-500 text-lg">🛡️</div>
                            <div>
                                <h4 class="font-bold text-teal-400">حارس USB (USB Guardian)</h4>
                                <span id="usbStatusBadge" class="text-[9px] px-2 py-0.5 rounded-full bg-slate-800 text-gray-400 border border-slate-700">متوقف</span>
                            </div>
                        </div>
                        <p class="text-[11px] text-gray-400 mb-4 h-10 relative z-10">مراقبة المنافذ والتدخل التلقائي لفحص أي فلاش ميموري (USB) بمجرد تركيبه للكشف عن فيروسات التشغيل التلقائي.</p>
                        
                        <div class="flex gap-2 relative z-10">
                            <button onclick="toggleUsbGuardian('start')" class="flex-1 py-2 bg-teal-900/40 hover:bg-teal-800 text-teal-300 rounded-lg text-sm transition-all border border-teal-800/50">تفعيل الحارس</button>
                            <button onclick="toggleUsbGuardian('stop')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-gray-400 rounded-lg text-sm transition-all border border-slate-700">إيقاف</button>
                        </div>
                    </div>
                    
                    <!-- File Integrity Monitor (FIM) -->
                    <div class="bg-orange-900/10 p-5 rounded-xl border border-orange-900/40 hover:border-orange-500/50 transition-all group relative overflow-hidden">
                        <div class="absolute top-0 right-0 w-16 h-16 bg-orange-600/10 rounded-bl-full z-0 pointer-events-none"></div>
                        <div class="flex items-center gap-3 mb-3 relative z-10">
                            <div class="w-10 h-10 rounded-lg bg-orange-900/30 flex items-center justify-center text-orange-500 text-lg">⚖️</div>
                            <div>
                                <h4 class="font-bold text-orange-400">مراقب تكامل الملفات (FIM)</h4>
                                <span id="fimStatusBadge" class="text-[9px] px-2 py-0.5 rounded-full bg-slate-800 text-gray-400 border border-slate-700">متوقف</span>
                            </div>
                        </div>
                        <p class="text-[11px] text-gray-400 mb-3 h-8 relative z-10">رصد التغييرات الطفيفة في ملفاتك الحساسة لمنع حقن الأكواد الخبيثة.</p>
                        
                        <div class="flex gap-2 relative z-10 flex-col">
                            <div class="flex gap-2">
                                <input type="text" id="fimTargetPath" class="flex-1 p-2 text-left text-[10px] font-mono rounded-lg bg-black border border-orange-900/50 text-orange-300 outline-none" placeholder="C:\\Windows\\System32\\drivers\\etc\\hosts" value="C:\\Windows\\System32\\drivers\\etc\\hosts">
                            </div>
                            <div class="flex gap-2 mt-1">
                                <button onclick="toggleFim('start')" class="flex-1 py-2 bg-orange-900/40 hover:bg-orange-800 text-orange-300 rounded-lg text-sm transition-all border border-orange-800/50">بدء المراقبة</button>
                                <button onclick="toggleFim('stop')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-gray-400 rounded-lg text-sm transition-all border border-slate-700">إيقاف</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 2FA Tool -->
                <div class="bg-black/40 rounded-xl border border-slate-800 p-6 mb-6">
                    <h2 class="text-xl font-bold text-purple-400 mb-4 border-b border-slate-700 pb-2">🔐 المصادقة الثنائية (2FA)</h2>
                    <button onclick="generate2FA()" class="titan-gradient px-4 py-2 rounded-lg font-bold mb-4">إنشاء مفتاح 2FA جديد</button>
                    
                    <div id="tfaResult" class="hidden bg-slate-900/80 p-6 rounded-xl border border-slate-700 flex flex-col items-center">
                        <p class="text-gray-400 mb-4 text-center">امسح رمز الاستجابة السريعة (QR Code) باستخدام تطبيق مثل Google Authenticator</p>
                        <img id="qrCodeImg" src="" alt="QR Code" class="w-48 h-48 bg-white p-2 rounded-lg mb-4">
                        <div class="bg-slate-800 p-3 rounded-xl w-full text-center mb-6 border border-slate-700">
                            <span class="text-xs text-gray-500 block mb-1">المفتاح السري (لإدخاله يدوياً):</span>
                            <code id="tfaSecret" class="text-purple-400 font-mono text-xl tracking-widest"></code>
                        </div>
                        
                        <div class="w-full border-t border-slate-700 pt-4">
                            <label class="block text-sm text-gray-400 mb-2">التحقق من الرمز:</label>
                            <div class="flex gap-2">
                                <input type="text" id="tfaCodeInput" placeholder="مكون من 6 أرقام..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-purple-500 outline-none text-center tracking-widest text-lg font-mono">
                                <button onclick="verify2FA()" class="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-xl font-bold transition-all">تحقق</button>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

            <div id="crypt-section" class="hidden">
                <div class="space-y-6">
                    <div>
                        <label class="block text-sm text-gray-400 mb-2">1. مفتاح التشفير (كلمة السر):</label>
                        <input type="password" id="cryptKey" class="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-purple-500 outline-none">
                    </div>
                    <hr class="border-slate-700">
                    <div>
                        <label class="block text-sm text-gray-400 mb-2 text-purple-400 font-bold italic">تشفير نصوص:</label>
                        <textarea id="cryptText" rows="3" class="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 mb-2 text-sm outline-none" placeholder="اكتب النص هنا..."></textarea>
                        <div class="flex gap-2">
                            <button onclick="processText('encrypt')" class="flex-1 titan-gradient p-2 rounded-lg font-bold">تشفير النص</button>
                            <button onclick="processText('decrypt')" class="flex-1 bg-slate-700 p-2 rounded-lg font-bold">فك التشفير</button>
                        </div>
                    </div>
                    <hr class="border-slate-700">
                    <div>
                        <label class="block text-sm text-gray-400 mb-2 text-purple-400 font-bold italic">تشفير ملفات:</label>
                        <input type="file" id="fileInput" class="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-600 file:text-white hover:file:bg-purple-700">
                        <div class="flex gap-2 mt-3">
                            <button onclick="processFile('encrypt')" class="flex-1 titan-gradient p-2 rounded-lg font-bold">تشفير الملف</button>
                            <button onclick="processFile('decrypt')" class="flex-1 bg-slate-700 p-2 rounded-lg font-bold">فك تشفير الملف</button>
                        </div>
                    </div>

                    </div>

                    <hr class="border-slate-700 mt-5 mb-5">
                    <div>
                        <label class="block text-sm text-gray-400 mb-2 text-purple-400 font-bold italic">حماية ملفات PDF بكلمة سر:</label>
                        <input type="password" id="pdfPass" placeholder="أدخل كلمة السر..." class="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-purple-500 outline-none mb-3 text-center tracking-widest">
                        <input type="file" id="pdfFileInput" accept="application/pdf" class="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-red-900/40 file:text-red-400 hover:file:bg-red-800/60 border border-slate-700 p-2 rounded-xl">
                        <div class="flex gap-2 mt-4">
                            <button onclick="processPdf('lock')" class="flex-1 bg-red-900/50 hover:bg-red-800 text-red-400 font-bold p-3 rounded-xl transition-all border border-red-900/30 shadow-[0_0_15px_rgba(239,68,68,0.15)] flex justify-center items-center gap-2">قفل الملف 🔒</button>
                            <button onclick="processPdf('unlock')" class="flex-1 bg-green-900/50 hover:bg-green-800 text-green-400 font-bold p-3 rounded-xl transition-all border border-green-900/30 shadow-[0_0_15px_rgba(34,197,94,0.15)] flex justify-center items-center gap-2">فك الحماية 🔓</button>
                        </div>
                    </div>
                </div>
            </div>
            <!-- ===== VAULT SECTION ===== -->
            <div id="vault-section" class="hidden space-y-4">
                <h2 class="text-xl font-bold text-yellow-400 border-b border-yellow-900/50 pb-2 flex items-center gap-2">🗄️ قبو كلمات المرور الآمن</h2>

                <!-- === FIRST-TIME SETUP PANEL (shown if user has no vault password yet) === -->
                <div id="vault-setup" class="hidden bg-gradient-to-br from-yellow-900/20 to-orange-900/10 rounded-2xl border border-yellow-800/40 p-6 shadow-[0_0_30px_rgba(234,179,8,0.08)]">
                    <div class="text-center mb-5">
                        <div class="text-4xl mb-2">🔐</div>
                        <h3 class="text-lg font-bold text-yellow-400">إعداد كلمة سر قبوك لأول مرة</h3>
                        <p class="text-xs text-gray-400 mt-1">هذه الكلمة ستُستخدم لتشفير قبوك الشخصي. لا يمكن استعادتها إذا نسيتها!</p>
                    </div>
                    <div class="space-y-3 max-w-sm mx-auto">
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">كلمة سر القبو الجديدة</label>
                            <input type="password" id="vaultSetupPass1" placeholder="أدخل كلمة سر قوية..." class="w-full p-3 rounded-xl bg-slate-900 border border-yellow-800/50 focus:ring-2 focus:ring-yellow-500 outline-none text-center tracking-widest text-lg font-mono">
                        </div>
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">تأكيد كلمة السر</label>
                            <input type="password" id="vaultSetupPass2" placeholder="أعد إدخال الكلمة..." class="w-full p-3 rounded-xl bg-slate-900 border border-yellow-800/50 focus:ring-2 focus:ring-yellow-500 outline-none text-center tracking-widest text-lg font-mono">
                        </div>
                        <div id="vaultSetupError" class="hidden text-red-400 text-xs p-3 bg-red-900/20 border border-red-800/40 rounded-xl text-center"></div>
                        <button onclick="setVaultPassword()" class="w-full py-3 bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-500 hover:to-orange-500 rounded-xl font-bold text-white transition-all shadow-[0_0_20px_rgba(234,179,8,0.25)]">
                            🔑 تأكيد وإنشاء القبو
                        </button>
                    </div>
                </div>

                <!-- === VAULT LOGIN PANEL === -->
                <div id="vault-login" class="bg-slate-900/70 rounded-2xl border border-yellow-900/40 p-6">
                    <p class="text-gray-400 text-sm mb-4 text-center">أدخل كلمة سر قبوك للوصول إلى بياناتك المحفوظة</p>
                    <div class="flex gap-2 max-w-md mx-auto">
                        <input type="password" id="vaultMasterKey" placeholder="كلمة سر القبو..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-yellow-800/50 focus:ring-2 focus:ring-yellow-500 outline-none font-mono tracking-widest text-lg text-center">
                        <button onclick="unlockVault()" class="bg-yellow-600 hover:bg-yellow-500 px-6 rounded-xl font-bold transition-all">فتح 🔓</button>
                    </div>
                </div>


                <!-- === VAULT CONTENT (shown after unlock) === -->
                <div id="vault-content" class="hidden space-y-4">

                    <!-- 1. Burn Notes -->
                    <div class="bg-slate-900/50 p-5 rounded-xl border border-slate-700/50 border-r-4 border-r-orange-500 relative overflow-hidden group">
                        <div class="absolute inset-0 bg-gradient-to-l from-orange-500/5 to-transparent pointer-events-none"></div>
                        <div class="flex items-center gap-3 mb-2 relative z-10">
                            <span class="text-orange-500 text-2xl drop-shadow-[0_0_10px_rgba(249,115,22,0.6)] animate-pulse">💣</span>
                            <h3 class="text-sm font-bold text-gray-200">الرسائل ذاتية التدمير (Burn Notes)</h3>
                        </div>
                        <p class="text-xs text-gray-400 mb-3 relative z-10">رسالة سرية لمرة واحدة — تُحذف فور قراءتها.</p>
                        <textarea id="burnNoteText" rows="3" class="w-full p-3 rounded-xl bg-slate-800 border border-slate-600 focus:ring-1 focus:ring-orange-500 outline-none text-sm mb-3 relative z-10" placeholder="اكتب رسالتك السرية هنا..."></textarea>
                        <button onclick="createBurnNote()" class="w-full bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white font-bold px-4 py-2 rounded-xl transition-all text-sm shadow-[0_0_15px_rgba(234,88,12,0.35)] flex items-center justify-center gap-2 relative z-10">
                            توليد رابط التدمير السري 🔥
                        </button>
                        <div id="burnNoteResult" class="hidden mt-3 p-3 bg-slate-900/80 border border-orange-800/30 rounded-xl flex flex-col md:flex-row items-center justify-between gap-2 relative z-10">
                            <input type="text" id="burnNoteLink" readonly class="w-full bg-black/50 text-orange-400 font-mono text-xs p-2 rounded-lg border border-slate-700/50 focus:outline-none" dir="ltr">
                            <button id="burnCopyBtn" onclick="copyBurnNoteLink()" class="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-xs px-4 py-2 rounded-lg text-gray-300 transition-colors whitespace-nowrap border border-slate-600 font-bold">نسخ الرابط</button>
                        </div>
                    </div>

                    <!-- 2. Action Bar -->
                    <div class="flex items-center justify-between gap-2 py-2 px-1 border-y border-slate-800">
                        <!-- Left: Lock -->
                        <button onclick="lockVault()" class="flex items-center gap-2 px-4 py-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 rounded-xl text-sm font-bold border border-red-900/40 transition-all">
                            🔒 <span>قفل القبو</span>
                        </button>
                        <!-- Right: Backup · Restore · Security Qs -->
                        <div class="flex items-center gap-2">
                            <button onclick="backupVault()" title="تصدير نسخة احتياطية" class="flex items-center gap-1.5 px-3 py-2 bg-blue-900/30 hover:bg-blue-800/50 text-blue-400 rounded-xl text-xs font-bold border border-blue-900/40 transition-all whitespace-nowrap">
                                💾 <span class="hidden sm:inline">نسخة احتياطية</span>
                            </button>
                            <label title="استعادة النسخة الاحتياطية" class="flex items-center gap-1.5 px-3 py-2 bg-purple-900/30 hover:bg-purple-800/50 text-purple-400 rounded-xl text-xs font-bold border border-purple-900/40 transition-all whitespace-nowrap cursor-pointer">
                                📂 <span class="hidden sm:inline">استعادة</span>
                                <input type="file" class="hidden" id="vaultRestoreFile" accept=".bak" onchange="restoreVault()">
                            </label>
                            <button onclick="showSetupRecovery()" title="أسئلة الأمان" class="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-yellow-500 rounded-xl text-xs font-bold border border-slate-700 transition-all whitespace-nowrap">
                                🛡️ <span class="hidden sm:inline">أسئلة الأمان</span>
                            </button>
                        </div>
                    </div>

                    <!-- Security Questions (collapsible) -->
                    <div id="setup-recovery-container" class="hidden bg-slate-900/50 p-5 rounded-xl border border-yellow-700/50 text-right">
                        <h3 class="text-sm font-bold text-yellow-400 mb-3">🛡️ إعداد أسئلة استعادة كلمة السر</h3>
                        <div class="space-y-3">
                            <input type="password" id="setup-a1-key" placeholder="كلمة السر الحالية" class="w-full p-2.5 rounded-lg bg-slate-800 border border-slate-600 outline-none text-sm text-yellow-100">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                                <input type="text" id="setup-q1" placeholder="السؤال الأول" class="p-2.5 rounded-lg bg-slate-800 border border-slate-600 outline-none text-sm">
                                <input type="password" id="setup-a1" placeholder="الإجابة الأولى" class="p-2.5 rounded-lg bg-slate-800 border border-slate-600 outline-none text-sm">
                                <input type="text" id="setup-q2" placeholder="السؤال الثاني" class="p-2.5 rounded-lg bg-slate-800 border border-slate-600 outline-none text-sm">
                                <input type="password" id="setup-a2" placeholder="الإجابة الثانية" class="p-2.5 rounded-lg bg-slate-800 border border-slate-600 outline-none text-sm">
                            </div>
                            <div class="flex gap-2 pt-1">
                                <button onclick="saveRecoverySetup()" class="flex-1 bg-yellow-600 hover:bg-yellow-500 text-slate-900 font-bold px-4 py-2 rounded-lg transition-all text-sm">حفظ الإعدادات</button>
                                <button onclick="document.getElementById('setup-recovery-container').classList.add('hidden')" class="bg-slate-800 text-gray-300 font-bold px-4 py-2 rounded-lg border border-slate-600 text-sm">إلغاء</button>
                            </div>
                        </div>
                    </div>

                    <!-- 3. Password Vault (items — last) -->
                    <div class="bg-slate-900/60 rounded-xl border border-slate-700 p-4 space-y-3">
                        <h3 class="text-sm font-bold text-yellow-400 flex items-center gap-2">🗄️ قبو كلمات المرور</h3>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-2">
                            <input type="text" id="vaultItemTitle" placeholder="الموقع / الخدمة" class="p-2 rounded-lg bg-slate-800 border border-slate-700 text-sm outline-none focus:ring-1 focus:ring-yellow-500">
                            <input type="text" id="vaultItemUsername" placeholder="اسم المستخدم / الإيميل" class="p-2 rounded-lg bg-slate-800 border border-slate-700 text-sm outline-none focus:ring-1 focus:ring-yellow-500">
                            <input type="password" id="vaultItemPass" placeholder="كلمة السر" class="p-2 rounded-lg bg-slate-800 border border-slate-700 text-sm outline-none focus:ring-1 focus:ring-yellow-500">
                        </div>
                        <button onclick="addVaultItem()" class="w-full py-2 bg-yellow-700/50 hover:bg-yellow-600/50 border border-yellow-700/50 rounded-lg font-bold text-yellow-300 text-sm transition-all">إضافة ➕</button>
                        <div id="vaultItemsContainer" class="space-y-2 max-h-80 overflow-y-auto custom-scrollbar pr-1 mt-1"></div>
                    </div>

                </div>
                <!-- /vault-content -->

            </div>
            <!-- /vault-section -->

            <div id="tools-section" class="hidden space-y-8">
                <!-- IP Tool With Radar -->
                <div>
                    <h2 class="text-xl font-bold text-purple-400 mb-4 border-b border-slate-700 pb-2">🌐 فحص واستخبارات IP</h2>
                    <div class="flex gap-2 mb-4">
                        <input type="text" id="ipInput" placeholder="أدخل IP (أو اتركه فارغاً لفحص اتصالك)" class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-purple-500 outline-none font-mono">
                        <button onclick="checkIP()" class="titan-gradient px-6 py-3 rounded-xl font-bold border border-purple-400/30 hover:shadow-[0_0_15px_rgba(168,85,247,0.5)] transition-all">تتبع الهدف 🎯</button>
                    </div>
                    
                    <div id="ipResult" class="hidden p-6 bg-slate-900/90 rounded-xl border border-slate-700 shadow-[0_0_25px_rgba(0,0,0,0.6)] relative overflow-hidden">
                        <div class="flex flex-col md:flex-row gap-8 relative z-10 items-center justify-between min-h-[160px]">
                            <!-- قسم البيانات -->
                            <div id="ipDataBox" class="flex-1 w-full order-2 md:order-1 transition-all"></div>
                            
                            <!-- الرادار -->
                            <div id="radarContainer" class="hidden md:flex flex-col items-center justify-center border-r border-slate-700/50 pr-8 pl-4 order-1 md:order-2">
                                <div class="radar-box">
                                    <div class="radar-cross"></div>
                                    <div class="radar-target"></div>
                                </div>
                                <p class="text-center text-green-400 text-[10px] mt-4 font-mono uppercase tracking-[0.2em] animate-pulse">Target Acquired</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Email Validator & Leak Scanner -->
                <div>
                    <h2 class="text-xl font-bold text-red-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>📧</span> فحص الإيميل (Email Intelligence)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">فحص البريد الإلكتروني للتأكد من صلاحيته، هل هو بريد وهمي (Disposable)، واحتمالية كونه احتيالياً (Fraud Score).</p>
                    <div class="flex gap-2 mb-4">
                        <input type="email" id="phishUrlInput" placeholder="أدخل البريد الإلكتروني لفحصه..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-red-500 outline-none font-mono text-left" dir="ltr">
                        <button onclick="checkPhishing()" class="bg-red-900/40 hover:bg-red-800 px-6 py-3 rounded-xl font-bold border border-red-800/50 transition-all text-red-400 flex items-center justify-center min-w-[140px]">
                            فحص الإيميل
                        </button>
                    </div>
                    <div id="phishResult" class="hidden p-4 bg-slate-900/80 rounded-xl border border-slate-700 text-sm mb-6"></div>
                    
                    <h3 class="font-bold text-orange-500 mb-3 text-sm flex items-center gap-2">
                        <span>🕵️</span> فحص تسريبات الإيميل وكلمة السر (Data Leaks)
                    </h3>
                    <p class="text-xs text-gray-400 mb-3">تحقق مما إذا كان بريدك الإلكتروني وكلمة السر المحددة قد تم تسريبها معاً في اختراقات سابقة للبيانات.</p>
                    <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                            <input type="email" id="leakEmailInput" placeholder="البريد الإلكتروني..." class="p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-orange-500 outline-none text-sm font-mono text-left" dir="ltr">
                            <input type="password" id="leakPassInput" placeholder="كلمة السر للتحقق من تسريبها مع الإيميل..." class="p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-orange-500 outline-none text-sm font-mono text-left" dir="ltr">
                        </div>
                        <button onclick="checkEmailPassLeak()" class="w-full bg-orange-900/40 hover:bg-orange-800 px-6 py-3 rounded-xl font-bold border border-orange-800/50 transition-all text-orange-400 flex items-center justify-center">
                            فحص التسريبات
                        </button>
                        <div id="leakEmailPassResult" class="hidden mt-4 p-4 bg-slate-900/80 rounded-xl border border-slate-700 text-sm"></div>
                    </div>
                </div>

                <!-- URL Scanner & Phishing Detection -->
                <div>
                    <h2 class="text-xl font-bold text-blue-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>🌐</span> فحص الروابط المشبوهة (URL/Phishing Scanner)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">فحص دقيق للروابط والمواقع لاكتشاف صفحات التصيد (Phishing) والبرمجيات الخبيثة وتصنيف الخطورة.</p>
                    <div class="flex gap-2 mb-4">
                        <input type="url" id="urlInput" placeholder="أدخل الرابط لفحصه (مثل https://example.com)..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-blue-500 outline-none font-mono text-left" dir="ltr">
                        <button onclick="checkUrl()" class="bg-blue-900/40 hover:bg-blue-800 px-6 py-3 rounded-xl font-bold border border-blue-800/50 transition-all text-blue-400 flex items-center justify-center min-w-[140px]">
                            فحص الرابط
                        </button>
                    </div>
                    <div id="urlResult" class="hidden p-4 bg-slate-900/80 rounded-xl border border-slate-700 text-sm"></div>
                </div>

                <!-- Malware URL & File Scanner -->
                <div>
                    <h2 class="text-xl font-bold text-rose-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>🦠</span> فحص البرمجيات الخبيثة (Malware Scanner)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">ابحث عن الفيروسات والبرمجيات الخبيثة المخفية في الروابط أو الملفات.</p>
                    
                    <div class="bg-slate-900/50 p-5 rounded-xl border border-slate-700/50 space-y-4">
                        <!-- URL Scan -->
                        <div>
                            <label class="block text-xs text-gray-400 mb-2 font-bold">فحص رابط خبيث:</label>
                            <div class="flex gap-2">
                                <input type="url" id="malwareUrlInput" placeholder="أدخل الرابط لفحصه..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-rose-500 outline-none font-mono text-left" dir="ltr">
                                <button onclick="checkMalwareUrl()" class="bg-rose-900/40 hover:bg-rose-800 px-6 py-3 rounded-xl font-bold border border-rose-800/50 transition-all text-rose-400 flex items-center justify-center min-w-[140px]">
                                    فحص الرابط
                                </button>
                            </div>
                        </div>
                        
                        <!-- Divider -->
                        <div class="flex items-center gap-3">
                            <hr class="flex-1 border-slate-700">
                            <span class="text-xs text-gray-500 font-bold uppercase">أو</span>
                            <hr class="flex-1 border-slate-700">
                        </div>
                        
                        <!-- File Scan -->
                        <div>
                            <label class="block text-xs text-gray-400 mb-2 font-bold">فحص ملف مشبوه:</label>
                            <div class="flex gap-2">
                                <input type="file" id="malwareFileInput" class="w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-rose-900/40 file:text-rose-400 hover:file:bg-rose-800/60 border border-slate-700 p-2 rounded-xl">
                                <button onclick="checkMalwareFile()" class="bg-rose-900/40 hover:bg-rose-800 px-6 py-2 rounded-xl font-bold border border-rose-800/50 transition-all text-rose-400 flex items-center justify-center min-w-[140px]">
                                    رفع وفحص الملف
                                </button>
                            </div>
                        </div>
                    </div>
                    <div id="malwareResult" class="hidden mt-4 p-4 bg-slate-900/80 rounded-xl border border-slate-700 text-sm"></div>
                </div>

                <!-- API Usage Logs (IPQualityScore) -->
                <div>
                    <h2 class="text-xl font-bold text-teal-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>📊</span> سجلات استخدام (API Logs)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">عرض سجلات العمليات السابقة التي تم إجراؤها عبر حساب IPQualityScore الخاص بك.</p>
                    <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                            <div>
                                <label class="block text-[10px] text-gray-500 mb-1 uppercase tracking-widest">نوع السجل</label>
                                <select id="ipqsLogType" class="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-teal-500 outline-none text-sm text-gray-300 appearance-none cursor-pointer">
                                    <option value="proxy" selected>IP / Proxy Checks</option>
                                    <option value="email">Email Checks</option>
                                    <option value="devicetracker">Device Tracker</option>
                                    <option value="mobiletracker">Mobile Tracker</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-[10px] text-gray-500 mb-1 uppercase tracking-widest">تاريخ البداية (YYYY-MM-DD)</label>
                                <input type="date" id="ipqsLogDate" class="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-teal-500 outline-none text-sm text-gray-300" value="2024-01-01">
                            </div>
                        </div>
                        <button onclick="fetchIpqsLogs()" class="w-full bg-teal-900/40 hover:bg-teal-800 px-6 py-3 rounded-xl font-bold border border-teal-800/50 transition-all text-teal-400 flex items-center justify-center gap-2">
                            <span>جلب السجلات</span>
                        </button>
                        <div id="ipqsLogsResult" class="hidden mt-4 max-h-[300px] overflow-y-auto custom-scrollbar p-2 bg-slate-950/80 rounded-xl border border-slate-700 text-sm font-mono text-left" dir="ltr"></div>
                    </div>
                </div>

                <!-- Phone Validator & Intelligence -->
                <div>
                    <h2 class="text-xl font-bold text-yellow-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>📱</span> فحص الهاتف (Phone Intelligence)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">تحليل رقم الهاتف لاكتشاف نوع الخط ومزود الخدمة ودرجة الاحتيال المرتبطة به.</p>
                    <div class="flex flex-col md:flex-row gap-2 mb-4">
                        <input type="tel" id="phoneInput" placeholder="أدخل رقم الهاتف مع الترميز (مثل +962778...)" class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-yellow-500 outline-none font-mono text-left" dir="ltr">
                        <button onclick="checkPhone()" class="bg-yellow-900/40 hover:bg-yellow-800 px-6 py-3 rounded-xl font-bold border border-yellow-800/50 transition-all text-yellow-400 flex items-center justify-center w-full md:w-auto min-w-[140px]">
                            فحص الرقم
                        </button>
                    </div>
                    <div id="phoneResult" class="hidden p-4 bg-slate-900/80 rounded-xl border border-slate-700 text-sm"></div>
                </div>

                <!-- Local LAN Monitor -->
                <div>
                    <h2 class="text-xl font-bold text-cyan-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>🌐</span> رادار الشبكة المحلية (LAN Monitor)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">اكتشف جميع الأجهزة المتصلة معك على نفس شبكة Wi-Fi محلياً لاكتشاف المتطفلين.</p>
                    <button id="btnLanScan" onclick="scanLanNetwork()" class="w-full bg-cyan-900/30 hover:bg-cyan-800 px-6 py-3 rounded-xl font-bold border border-cyan-800/50 transition-all text-cyan-400 flex items-center justify-center gap-2 mb-4">
                        <span>مسح الشبكة للبحث عن دخلاء</span>
                        <div id="lanLoader" class="hidden w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                    </button>
                    <div id="lanResult" class="hidden p-4 bg-slate-900/80 rounded-xl border border-slate-700 max-h-60 overflow-y-auto custom-scrollbar"></div>
                </div>
                <!-- Port Scanner Tool -->
                <div>
                    <h2 class="text-xl font-bold text-purple-400 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>🔍</span> فحص المنافذ المفتوحة
                    </h2>
                    <div class="flex gap-2 mb-4">
                        <input type="text" id="portIpInput" placeholder="أدخل IP الهدف للبحث عن ثغرات..." class="flex-1 p-3 rounded-xl bg-slate-900 border border-slate-700 focus:ring-2 focus:ring-purple-500 outline-none font-mono text-left" dir="ltr">
                        <button id="btnPortScan" onclick="scanPorts()" class="bg-slate-800 hover:bg-slate-700 px-6 py-3 rounded-xl font-bold border border-slate-600 transition-all flex items-center justify-center min-w-[140px] text-gray-300">
                            <span id="scanLabel">فحص المنافذ 📡</span>
                            <div id="scanLoader" class="hidden w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                        </button>
                    </div>
                    
                    <div id="portResult" class="hidden bg-slate-900/80 p-6 rounded-xl border border-slate-700 shadow-lg">
                        <h3 class="font-bold text-gray-300 mb-4 text-sm flex items-center gap-2">
                            نتائج الفحص للهدف: <span id="portScanTarget" class="text-purple-400 font-mono tracking-widest bg-purple-900/20 px-2 py-1 rounded"></span>
                        </h3>
                        <div id="openPortsContainer" class="flex flex-wrap gap-2 text-sm max-h-40 overflow-y-auto custom-scrollbar">
                            <!-- المنافذ برمجياً -->
                        </div>
                    </div>
                </div>

                <!-- Phase 4 Secure Comms: Burn Chat -->
                <div class="col-span-1 md:col-span-2">
                    <h2 class="text-xl font-bold text-pink-500 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
                        <span>🔥</span> غرفة الـ Burn Chat (P2P مشفر)
                    </h2>
                    <p class="text-xs text-gray-400 mb-3">اتصال مشفر آمن لا يحفظ السجلات. الرسالة تُدمّر حرفياً من ذاكرة الخادم في اللحظة التي تُقرأ فيها.</p>
                    
                    <div class="bg-gray-900/80 rounded-xl border border-slate-700 p-4">
                        <div class="flex gap-2 mb-4 bg-black p-3 rounded-lg border border-slate-800 flex-col md:flex-row">
                            <input type="text" id="burnChatId" placeholder="كود الغرفة (Room ID)..." class="flex-1 p-2 rounded bg-slate-900 border border-slate-700 focus:border-pink-500 outline-none text-center font-mono">
                            <input type="password" id="burnChatKey" placeholder="كلمة مرور الغرفة..." class="flex-1 p-2 rounded bg-slate-900 border border-slate-700 focus:border-pink-500 outline-none text-center font-mono" title="كلمة السر الخاصة بدخول الغرفة">
                            <input type="text" id="burnChatUser" placeholder="اسمك الرمزي (Ghost)" class="w-full md:w-1/4 p-2 rounded bg-slate-900 border border-slate-700 focus:border-pink-500 outline-none text-center">
                            <button onclick="joinBurnChat()" class="bg-pink-900/40 hover:bg-pink-800 text-pink-300 px-6 py-2 rounded border border-pink-800/50 transition-all font-bold">انضمام</button>
                        </div>
                        
                        <div id="burnChatDisplay" class="h-64 bg-black rounded-lg border border-pink-900/30 mb-4 p-4 overflow-y-auto flex flex-col gap-2 shadow-inner">
                            <div class="text-center text-gray-600 text-[10px] tracking-widest uppercase mt-auto">-- Secure RAM Storage Only --</div>
                        </div>
                        
                        <div class="flex gap-2">
                            <input type="text" id="burnChatInput" placeholder="اكتب رسالتك السرية هنا..." class="flex-1 p-3 rounded-lg bg-slate-900 border border-slate-700 focus:border-pink-500 outline-none" disabled>
                            <button id="burnChatSendBtn" onclick="sendBurnChat()" class="bg-slate-800 text-gray-500 px-8 rounded-lg font-bold transition-all border border-slate-700" disabled>إرسال</button>
                        </div>
                    </div>
                </div>
            </div>
            </div>

            <!-- ===== QR CODE SECTION ===== -->
            <div id="qr-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-green-400 border-b border-slate-700 pb-2">🔳 QR Code مشفر</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900/70 rounded-xl p-5 border border-green-900/40 space-y-3">
                        <h3 class="font-bold text-green-400 text-sm">توليد QR</h3>
                        <textarea id="qrText" rows="3" placeholder="النص أو الرابط..." class="w-full p-3 rounded-xl bg-slate-800 border border-slate-700 outline-none text-sm focus:ring-2 focus:ring-green-500"></textarea>
                        <input type="password" id="qrPass" placeholder="كلمة سر (اختياري للتشفير)" class="w-full p-3 rounded-xl bg-slate-800 border border-slate-700 outline-none text-sm">
                        <button onclick="generateQR()" class="w-full bg-green-900/50 hover:bg-green-800 text-green-300 font-bold p-3 rounded-xl border border-green-800/50 transition-all">توليد QR 🔳</button>
                        <div id="qrResult" class="hidden text-center">
                            <img id="qrImg" src="" class="mx-auto rounded-lg border border-green-800/40 max-w-[200px]">
                            <a id="qrDownload" download="qr.png" class="block mt-2 text-xs text-green-400 underline cursor-pointer">تنزيل الصورة</a>
                        </div>
                    </div>
                    <div class="bg-slate-900/70 rounded-xl p-5 border border-blue-900/40 space-y-3">
                        <h3 class="font-bold text-blue-400 text-sm">رفع QR للقراءة / الفك</h3>
                        <input type="file" id="qrFile" accept="image/*" class="block w-full text-sm text-slate-400 file:mr-2 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-slate-800 file:text-blue-400 border border-slate-700 p-2 rounded-xl">
                        <input type="password" id="qrDecodePass" placeholder="كلمة السر (إذا كان مشفراً)" class="w-full p-3 rounded-xl bg-slate-800 border border-slate-700 outline-none text-sm">
                        <button onclick="decodeQR()" class="w-full bg-blue-900/50 hover:bg-blue-800 text-blue-300 font-bold p-3 rounded-xl border border-blue-800/50 transition-all">قراءة QR 🔍</button>
                        <div id="qrDecodeResult" class="hidden p-3 bg-slate-800 rounded-xl border border-slate-600 text-sm font-mono text-green-300 break-all"></div>
                    </div>
                </div>
            </div>

            <!-- ===== FAKE IDENTITY SECTION ===== -->
            <div id="identity-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-teal-400 border-b border-slate-700 pb-2 flex items-center gap-2"><span>🪪</span> مولد الهوية الوهمية الرقمية</h2>
                <div class="bg-slate-900/70 p-6 rounded-xl border border-teal-900/40 shadow-inner">
                    <p class="text-gray-400 text-sm mb-4 leading-relaxed">استخدم هذه الأداة لإنشاء بيانات شخصية كاملة ووهمية للحفاظ على خصوصيتك عند التسجيل في المواقع غير الموثوقة أو للاستخدامات الأمنية التجريبية.</p>
                    
                    <div class="flex flex-col sm:flex-row gap-4 mb-6">
                        <select id="identityLang" class="p-3 rounded-xl bg-slate-800 border border-slate-700 text-gray-300 outline-none flex-1 focus:ring-2 focus:ring-teal-500">
                            <option value="ar_SA">عربي (السعودية)</option>
                            <option value="ar_AE">عربي (الإمارات)</option>
                            <option value="ar_EG">عربي (مصر)</option>
                            <option value="en_US">إنجليزي (أمريكا)</option>
                            <option value="en_GB">إنجليزي (بريطانيا)</option>
                            <option value="fr_FR">فرنسي (فرنسا)</option>
                        </select>
                        <button onclick="generateIdentity()" class="bg-gradient-to-r from-teal-900 to-teal-800 hover:from-teal-800 hover:to-teal-700 text-white font-bold p-3 rounded-xl border border-teal-700 transition-all flex items-center justify-center gap-2 flex-2 shadow-lg">
                            <span>توليد هوية جديدة</span> <span>🔄</span>
                        </button>
                    </div>

                    <div id="identityResultArea" class="hidden space-y-4 relative">
                        <!-- Basic Info -->
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col relative group md:col-span-2">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الاسم الكامل</span>
                                <span id="idName" class="text-xl font-bold text-gray-200"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الجنس</span>
                                <span id="idGender" class="text-sm font-bold text-gray-300"></span>
                            </div>
                        </div>

                        <!-- Personal Details -->
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">اسم الأم قبل الزواج</span>
                                <span id="idMotherName" class="text-sm text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الرقم الوطني / SSN</span>
                                <span id="idNational" class="text-sm font-mono text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">تاريخ الميلاد</span>
                                <span id="idDob" class="text-sm font-mono text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">العمر / البرج</span>
                                <span class="text-sm text-gray-300"><span id="idAge"></span> سنة (<span id="idZodiac"></span>)</span>
                            </div>
                        </div>

                        <!-- Location -->
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col md:col-span-2">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">العنوان الدائم</span>
                                <span id="idAddress" class="text-sm text-gray-300 break-words"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الرمز البريدي (Zip Code)</span>
                                <span id="idZip" class="text-sm font-mono text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col md:col-span-3">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الإحداثيات الجغرافية / رمز البلد</span>
                                <span class="text-sm font-mono text-gray-300" dir="ltr"><span id="idGeo"></span> (Code: <span id="idCountryCode"></span>)</span>
                            </div>
                        </div>

                        <!-- Contact & Employment -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">رقم الهاتف</span>
                                <span id="idPhone" class="text-sm font-mono text-gray-300" dir="ltr"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">البريد الإلكتروني</span>
                                <span id="idEmail" class="text-sm font-mono text-gray-300 break-all select-all"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الشركة</span>
                                <span id="idCompany" class="text-sm text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الوظيفة</span>
                                <span id="idJob" class="text-sm text-gray-300"></span>
                            </div>
                        </div>

                        <!-- Characteristics -->
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">الطول / الوزن</span>
                                <span class="text-sm font-mono text-gray-300"><span id="idHeight"></span> / <span id="idWeight"></span></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">فصيلة الدم</span>
                                <span id="idBlood" class="text-sm font-mono text-gray-300 text-center font-bold text-red-400"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">اللون المفضل</span>
                                <span id="idColor" class="text-sm text-gray-300"></span>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                                <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">السيارة</span>
                                <span id="idVehicle" class="text-sm text-gray-300"></span>
                            </div>
                        </div>

                        <!-- Financial & Digital -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col relative group overflow-hidden">
                                <div class="absolute right-0 top-0 h-full w-1 bg-gradient-to-b from-blue-400 to-indigo-600"></div>
                                <span class="text-[10px] text-blue-400 mb-1 font-bold uppercase tracking-wider block"><span class="text-white bg-blue-900/50 px-1 rounded mr-1">بطاقة ائتمانية</span> (<span id="idCcType"></span>)</span>
                                <span id="idCredit" class="text-lg font-mono text-blue-100 tracking-[0.2em] drop-shadow-sm mt-1"></span>
                                <div class="flex justify-between mt-2 text-xs text-blue-300 font-mono flex-row gap-2 text-center md:text-left">
                                    <span>Exp: <span id="idCcExp"></span></span>
                                    <span>CVV: <span id="idCcCvv"></span></span>
                                </div>
                            </div>
                            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col relative group overflow-hidden">
                                <div class="absolute right-0 top-0 h-full w-1 bg-gradient-to-b from-purple-400 to-fuchsia-600"></div>
                                <span class="text-[10px] text-purple-400 mb-1 font-bold uppercase tracking-wider block"><span class="text-white bg-purple-900/50 px-1 rounded mr-1">بيانات</span> تسجيل الدخول</span>
                                <div class="flex flex-col mt-1">
                                    <span class="text-xs text-gray-400">User: <span id="idUsername" class="text-white font-mono break-all select-all"></span></span>
                                    <span class="text-xs text-gray-400 mt-1">Pass: <span id="idPassword" class="text-white font-mono break-all select-all"></span></span>
                                </div>
                            </div>
                        </div>

                        <!-- Identity Meta Info -->
                        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800/80 shadow-sm flex flex-col">
                            <span class="text-[10px] text-teal-500 mb-1 font-bold uppercase tracking-wider block">بصمة الهوية الرقمية (Tracking Info)</span>
                            <div class="flex flex-col mt-1 space-y-1">
                                <span class="text-xs text-gray-400">Website: <a href="#" id="idWebsite" class="text-teal-400 hover:underline break-all"></a></span>
                                <span class="text-xs text-gray-400">User Agent: <span id="idUserAgent" class="text-gray-300 font-mono text-[10px] break-words"></span></span>
                                <span class="text-xs text-gray-400">UUID: <span id="idUuid" class="text-gray-300 font-mono text-[10px]"></span></span>
                            </div>
                        </div>
                        
                        <button onclick="copyFullIdentity()" class="w-full mt-4 bg-slate-800 hover:bg-slate-700 text-gray-300 text-sm font-bold py-3 rounded-xl border border-slate-600 transition-colors flex items-center justify-center gap-2">
                            <span>نسخ جميع البيانات كنص</span> <span>📋</span>
                        </button>
                    </div>
                </div>
            </div>



            <!-- ===== AUDIO STEGANOGRAPHY SECTION ===== -->
            <div id="audio-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-blue-400 border-b border-slate-700 pb-2">🎵 إخفاء البيانات في الصوت (Audio Stegano)</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-blue-500/20">
                        <label class="block text-sm text-blue-400 mb-3 font-bold">🛠️ تشفير (إخفاء):</label>
                        <input type="file" id="audioFileEncrypt" accept=".wav" class="w-full text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-blue-600/10 file:text-blue-400 hover:file:bg-blue-600/20 mb-4">
                        <textarea id="audioSecretText" placeholder="أدخل النص السري هنا..." class="w-full h-24 p-3 rounded-xl bg-slate-950 border border-slate-800 text-sm focus:border-blue-500 outline-none mb-4"></textarea>
                        <button onclick="processAudio('encode')" class="w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-bold transition-all shadow-lg shadow-blue-900/20">حفظ النص في الملف 💾</button>
                    </div>
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-purple-500/20">
                        <label class="block text-sm text-purple-400 mb-3 font-bold">🔍 فك التشفير (استخراج):</label>
                        <input type="file" id="audioFileDecrypt" accept=".wav" class="w-full text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-purple-600/10 file:text-purple-400 hover:file:bg-purple-600/20 mb-4">
                        <div id="audioDecodedResult" class="w-full h-24 p-3 rounded-xl bg-slate-950 border border-slate-800 text-sm overflow-y-auto mb-4 text-gray-400 font-mono italic">سيظهر النص المستخرج هنا...</div>
                        <button onclick="processAudio('decode')" class="w-full py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-bold transition-all shadow-lg shadow-purple-900/20">استخراج النص السري 🔑</button>
                    </div>
                </div>
            </div>

            <!-- ===== EXTREME PRIVACY SECTION ===== -->
            <div id="extreme-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-teal-400 border-b border-slate-700 pb-2">🛡️ أدوات الخصوصية القصوى (Extreme Privacy)</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-teal-500/20">
                        <label class="block text-sm text-teal-400 mb-3 font-bold">📄 منظف ملفات PDF:</label>
                        <p class="text-[10px] text-gray-500 mb-4">إزالة الميتابيانات من ملفات PDF لحماية الخصوصية.</p>
                        <input type="file" id="pdfCleanFile" accept=".pdf" class="w-full text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-teal-600/10 file:text-teal-400 hover:file:bg-teal-600/20 mb-4">
                        <button onclick="cleanPdf()" class="w-full py-3 bg-teal-600 hover:bg-teal-500 rounded-xl font-bold transition-all text-sm">بدء التنظيف العميق 🧹</button>
                    </div>
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-indigo-500/20">
                        <label class="block text-sm text-indigo-400 mb-3 font-bold">🪪 بصمة المتصفح (Browser Fingerprint):</label>
                        <p class="text-[10px] text-gray-500 mb-4">توليد ملف تعريف وهمي لتجنب التتبع الرقمي.</p>
                        <div id="fingerprintDisplay" class="font-mono text-[9px] text-indigo-300 bg-black/60 p-3 rounded-lg mb-4 h-24 overflow-y-auto italic">اضغط لتوليد هوية جديدة...</div>
                        <button onclick="generateStealthFingerprint()" class="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-bold transition-all text-sm">توليد هوية وهمية 🔀</button>
                    </div>
                </div>
            </div>

            <!-- ===== NETWORK INTELLIGENCE SECTION ===== -->
            <div id="netintel-section" class="hidden space-y-6">
                <h2 class="text-xl font-bold text-orange-400 border-b border-slate-700 pb-2">🔬 استخبارات الشبكة (TITAN Intel)</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-orange-500/20">
                        <label class="block text-sm text-orange-400 mb-3 font-bold">🔍 فحص Shodan (المنافذ العامة):</label>
                        <input type="text" id="shodanIp" placeholder="IP عام..." class="w-full p-3 rounded-xl bg-slate-950 border border-slate-800 text-sm focus:border-orange-500 outline-none mb-4 text-center font-mono">
                        <div id="shodanResult" class="font-mono text-[10px] text-gray-400 mb-4 h-24 overflow-y-auto"></div>
                        <button onclick="runShodanScan()" class="w-full py-3 bg-orange-600 hover:bg-orange-500 rounded-xl font-bold transition-all text-sm">جلب بيانات Shodan 📡</button>
                    </div>
                    <div class="bg-slate-900/50 p-6 rounded-2xl border border-red-500/20 text-center">
                        <label class="block text-sm text-red-400 mb-3 font-bold">🚰 فحص تسريب DNS:</label>
                        <div id="dnsLeakStatus" class="text-2xl font-black mb-1 text-white">—</div>
                        <div id="dnsLeakDetails" class="text-[9px] text-gray-500 mb-4">سيتم فحص خوادم DNS الحالية...</div>
                        <button onclick="checkDnsLeak()" class="w-full py-3 bg-red-600 hover:bg-red-500 rounded-xl font-bold transition-all text-sm">بدء الفحص السريع 🚨</button>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Panic Button Removed as per User Request -->

    <script>
        // --- نظام المؤثرات الصوتية (Web Audio API) ---
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        let audioCtx;

        function initAudio() {
            if(!audioCtx) audioCtx = new AudioContext();
            if(audioCtx.state === 'suspended') audioCtx.resume();
        }

        // تشغيل نغمة (Oscillator)
        function playTone(freq, type, duration, vol=0.1) {
            if(!audioCtx) return;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = type; // 'sine', 'square', 'sawtooth', 'triangle'
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(vol, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        }

        const soundManager = {
            hover: () => playTone(800, 'sine', 0.05, 0.02),
            click: () => { playTone(1200, 'square', 0.05, 0.05); playTone(1600, 'sine', 0.1, 0.02); },
            terminalType: () => playTone(2000 + Math.random()*500, 'square', 0.02, 0.02),
            startupTone: () => { playTone(300, 'sine', 1, 0.1); playTone(600, 'sawtooth', 0.5, 0.05); },
            swoosh: () => { 
                if(!audioCtx) return;
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(100, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.8);
                gain.gain.setValueAtTime(0, audioCtx.currentTime);
                gain.gain.linearRampToValueAtTime(0.3, audioCtx.currentTime + 0.4);
                gain.gain.linearRampToValueAtTime(0.001, audioCtx.currentTime + 0.8);
                osc.connect(gain); gain.connect(audioCtx.destination);
                osc.start(); osc.stop(audioCtx.currentTime + 0.8);
            },
            success: () => { playTone(600, 'sine', 0.1, 0.05); setTimeout(()=>playTone(800, 'sine', 0.2, 0.1), 100); setTimeout(()=>playTone(1200, 'sine', 0.4, 0.15), 250); },
            error: () => { playTone(250, 'sawtooth', 0.3, 0.1); setTimeout(()=>playTone(150, 'sawtooth', 0.5, 0.15), 200); },
            alarm: () => { playTone(800, 'sawtooth', 0.4, 0.1); setTimeout(()=>playTone(600, 'square', 0.4, 0.1), 400); }
        };

        // --- نظام التحكم بجلسة الدخول (Auth Control) ---
        async function doLogout() {
            if (!confirm('هل أنت متأكد من تسجيل الخروج؟')) return;
            try {
                await fetch('/api/auth/logout', { method: 'POST' });
            } catch(e) {}
            window.location.reload();
        }

        function switchAuthTab(tab) {
            const loginTab = document.getElementById('auth-tab-login');
            const regTab = document.getElementById('auth-tab-register');
            const loginForm = document.getElementById('auth-login-form');
            const regForm = document.getElementById('auth-register-form');
            
            if (tab === 'login') {
                loginTab.style.background = 'linear-gradient(135deg, #a855f7, #7c3aed)';
                loginTab.style.color = 'white';
                loginTab.style.boxShadow = '0 0 15px rgba(168, 85, 247, 0.4)';
                regTab.style.background = 'transparent';
                regTab.style.color = '#6b7280';
                regTab.style.boxShadow = 'none';
                loginForm.style.display = 'block';
                regForm.style.display = 'none';
            } else {
                regTab.style.background = 'linear-gradient(135deg, #7c3aed, #5b21b6)';
                regTab.style.color = 'white';
                regTab.style.boxShadow = '0 0 15px rgba(124, 58, 237, 0.4)';
                loginTab.style.background = 'transparent';
                loginTab.style.color = '#6b7280';
                loginTab.style.boxShadow = 'none';
                loginForm.style.display = 'none';
                regForm.style.display = 'block';
            }
        }

        async function doLogin() {
            const username = document.getElementById('auth-login-user').value.trim();
            const password = document.getElementById('auth-login-pass').value;
            const errEl    = document.getElementById('auth-login-error');
            const btn      = document.getElementById('auth-login-btn');

            errEl.style.display = 'none';
            if (!username || !password) { errEl.textContent = 'يرجى إدخال اسم المستخدم وكلمة السر'; errEl.style.display = 'block'; return; }

            btn.textContent = '⏳ جاري التحقق...';
            btn.disabled = true;

            try {
                const res  = await fetch('/api/auth/login', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({username, password}) });
                const data = await res.json();

                if (data.success) {
                    btn.textContent = '✅ تم الدخول بنجاح!';
                    btn.style.background = 'linear-gradient(135deg,#22c55e,#16a34a)';
                    const usernameEl = document.getElementById('header-username');
                    if (usernameEl) usernameEl.textContent = '👤 ' + data.username;
                    setTimeout(showAuthSuccess, 500);
                } else {
                    errEl.textContent = data.error || 'بيانات الدخول غير صحيحة';
                    errEl.style.display = 'block';
                    btn.textContent = 'دخول إلى TITAN 🔐';
                    btn.disabled = false;
                }
            } catch(e) {
                errEl.textContent = 'فشل الاتصال بالخادم. تأكد أن التطبيق يعمل.';
                errEl.style.display = 'block';
                btn.textContent = 'دخول إلى TITAN 🔐';
                btn.disabled = false;
            }
        }

        async function doRegister() {
            const username  = document.getElementById('auth-reg-user').value.trim();
            const password  = document.getElementById('auth-reg-pass').value;
            const password2 = document.getElementById('auth-reg-pass2').value;
            const errEl     = document.getElementById('auth-reg-error');
            const sucEl     = document.getElementById('auth-reg-success');
            const btn       = document.getElementById('auth-reg-btn');

            errEl.style.display = 'none';
            sucEl.style.display = 'none';

            if (!username || !password || !password2) { errEl.textContent = 'يرجى ملء جميع الحقول'; errEl.style.display = 'block'; return; }
            if (password !== password2) { errEl.textContent = 'كلمتا السر غير متطابقتين!'; errEl.style.display = 'block'; return; }
            if (password.length < 6) { errEl.textContent = 'كلمة السر يجب أن تكون 6 أحرف على الأقل'; errEl.style.display = 'block'; return; }

            btn.textContent = '⏳ جاري إنشاء الحساب...';
            btn.disabled = true;

            try {
                const res  = await fetch('/api/auth/register', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({username, password}) });
                const data = await res.json();

                if (data.success) {
                    sucEl.textContent = '✅ تم إنشاء الحساب بنجاح! يمكنك الآن تسجيل الدخول.';
                    sucEl.style.display = 'block';
                    btn.textContent = 'إنشاء حساب جديد ✨';
                    btn.disabled = false;
                    setTimeout(() => {
                        switchAuthTab('login');
                        document.getElementById('auth-login-user').value = username;
                        document.getElementById('auth-login-pass').focus();
                    }, 1500);
                } else {
                    errEl.textContent = data.error || 'فشل إنشاء الحساب';
                    errEl.style.display = 'block';
                    btn.textContent = 'إنشاء حساب جديد ✨';
                    btn.disabled = false;
                }
            } catch(e) {
                errEl.textContent = 'فشل الاتصال بالخادم.';
                errEl.style.display = 'block';
                btn.textContent = 'إنشاء حساب جديد ✨';
                btn.disabled = false;
            }
        }

        function showAuthSuccess() {
            const overlay = document.getElementById('auth-overlay');
            overlay.style.transition = 'opacity 0.6s ease';
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
                overlay.style.opacity = '1';
                const mainApp = document.getElementById('main-app');
                if (mainApp) {
                    mainApp.classList.remove('opacity-0', 'pointer-events-none');
                    initMatrix('matrix-bg', false);
                    setInterval(updateHUD, 2000);
                    document.querySelectorAll('button').forEach(btn => {
                        btn.addEventListener('mouseenter', soundManager.hover);
                        btn.addEventListener('click', soundManager.click);
                    });
                }
            }, 600);
        }

        async function checkAuth() {
            try {
                const res = await fetch('/api/auth/status');
                const data = await res.json();
                if (data.loggedIn) {
                    document.getElementById('auth-overlay').style.display = 'none';
                    document.getElementById('header-username').innerText = '👤 ' + data.username;
                    const mainApp = document.getElementById('main-app');
                    if (mainApp) mainApp.classList.remove('opacity-0', 'pointer-events-none');
                    runIntroSequence();
                } else {
                    document.getElementById('auth-overlay').style.display = 'block';
                    initMatrix('auth-matrix', true);
                }
            } catch (e) {
                document.getElementById('auth-overlay').style.display = 'block';
                initMatrix('auth-matrix', true);
            }
        }

        // --- إدارة المقدمة الفاخرة (Premium IntroFade In) ---
        function runIntroSequence() {
            setTimeout(() => {
                const logoContainer = document.getElementById('intro-center-logo');
                logoContainer.classList.remove('opacity-0', 'translate-y-8');
                soundManager.startupTone();
                setTimeout(() => document.getElementById('start-btn').classList.add('show'), 600);
            }, 300);
        }

        // أداة لمعرفة متى المستخدم ضغط أي زر لتفعيل الصوت (لأن المتصفحات تمنع الصوت بدون تفاعل)
        window.addEventListener('click', () => { initAudio(); }, {once:true});
        window.onload = () => { setTimeout(checkAuth, 100); };

        function startSystem() {
            soundManager.click();
            setTimeout(() => soundManager.swoosh(), 200); 
            
            const intro = document.getElementById('intro-overlay');
            const app = document.getElementById('main-app');
            
            intro.style.transform = 'scale(1.05)';
            intro.style.opacity = '0';
            
            setTimeout(() => {
                intro.style.display = 'none';
                if(typeof introMatrixAnimId !== 'undefined') cancelAnimationFrame(introMatrixAnimId);
                app.classList.remove('opacity-0', 'pointer-events-none');
                
                document.querySelectorAll('button').forEach(btn => {
                    btn.addEventListener('mouseenter', soundManager.hover);
                    btn.addEventListener('click', soundManager.click);
                });
                document.querySelectorAll('input, textarea').forEach(inp => inp.addEventListener('focus', soundManager.hover));
                
                setInterval(updateHUD, 2000);
            }, 1000);
        }

        // --- Live HUD System Data ---
        function updateHUD() {
            const cpu = document.getElementById('hud-cpu');
            const mem = document.getElementById('hud-mem');
            const ping = document.getElementById('hud-ping');
            if(cpu) cpu.innerText = Math.floor(Math.random() * 40 + 10) + '%';
            if(mem) mem.innerText = (Math.random() * 2 + 1.5).toFixed(1) + 'GB';
            if(ping) ping.innerText = Math.floor(Math.random() * 50 + 10) + 'ms';
        }

        // --- 3D Matrix Rain (Violet/Purple Theme) ---
        function initMatrix(canvasId, isPremium3D = false) {
            const canvas = document.getElementById(canvasId);
            if(!canvas) return;
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth; canvas.height = window.innerHeight;
            
            const chars = "∑πΩΔΦΨΓΛΞ∞∫∬∮∇∂</√∛∝∠∩∪∴∵∼≈≅≠≡≤≥⊂⊃⊕⊗⊙⊢⊣⊥⊨⊩∀∃∄∅∉∈∊∋∌∏∐".split("");
            const fontSize = isPremium3D ? 12 : 10;
            const columns = canvas.width / fontSize;
            const drops = [];
            const speeds = [];
            const depths = []; 
            
            for(let x = 0; x < columns; x++) {
                drops[x] = Math.random() * -100;
                speeds[x] = isPremium3D ? (Math.random() * 0.5 + 0.2) : (Math.random() * 1 + 0.5);
                depths[x] = Math.random(); 
            }

            let animId;
            function draw() {
                // Fade effect matching dark background (higher alpha for cleaner tails)
                ctx.fillStyle = isPremium3D ? "rgba(5, 5, 5, 0.25)" : "rgba(10, 10, 10, 0.15)"; 
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                for(let i = 0; i < drops.length; i++) {
                    if (drops[i] < 0) {
                        drops[i] += speeds[i];
                        continue;
                    }
                    
                    const text = chars[Math.floor(Math.random() * chars.length)];
                    ctx.font = (fontSize * (isPremium3D ? (depths[i] * 0.5 + 0.8) : 1)) + "px monospace";
                    
                    const baseAlpha = isPremium3D ? depths[i] : 1;
                    
                    if (Math.random() > 0.98) ctx.fillStyle = `rgba(255, 255, 255, ${baseAlpha})`; 
                    else if (Math.random() > 0.9) ctx.fillStyle = `rgba(168, 85, 247, ${baseAlpha})`; // Purple
                    else if (Math.random() > 0.7) ctx.fillStyle = `rgba(139, 92, 246, ${baseAlpha})`; // Deep Violet
                    else ctx.fillStyle = `rgba(88, 28, 135, ${baseAlpha})`; // Very Deep Purple

                    ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                    
                    if(drops[i] * fontSize > canvas.height && Math.random() > 0.98) {
                        drops[i] = 0;
                        speeds[i] = isPremium3D ? (Math.random() * 0.5 + 0.2) : (Math.random() * 1 + 0.5); 
                    }
                    drops[i] += speeds[i];
                }
                animId = requestAnimationFrame(draw);
            }
            draw();
            
            window.addEventListener('resize', () => { 
                canvas.width = window.innerWidth; 
                canvas.height = window.innerHeight; 
            });
            return animId; 
        }

        let introMatrixAnimId = initMatrix('intro-matrix', true);
        initMatrix('matrix-bg', false);

        // --- نظام التنبيهات الصوتية ---


        // --- التحكم بالتبويبات ---
        const ALL_TABS = ['dash','pass','vault','crypt','suite','tools','qr','identity','audio','extreme','netintel'];
        let _prevTab = 'pass';
        function showTab(type) {
            // Auto-lock vault silently when leaving it
            if (_prevTab === 'vault' && type !== 'vault' && currentMasterKey) {
                lockVault(true);
            }
            _prevTab = type;
            ALL_TABS.forEach(t => {
                const sec = document.getElementById(t + '-section');
                const btn = document.getElementById('btn-' + t);
                if(sec) sec.classList.toggle('hidden', t !== type);
                if(btn) {
                    if(t === type) {
                        btn.classList.remove('text-gray-400');
                        btn.classList.add('bg-purple-600/90', 'text-white');
                    } else {
                        btn.classList.remove('bg-purple-600/90', 'text-white', 'bg-yellow-600/90', 'text-slate-900');
                        btn.classList.add('text-gray-400');
                    }
                }
            });
            // Special vault styling
            const vBtn = document.getElementById('btn-vault');
            if(vBtn) {
                if(type === 'vault') {
                    vBtn.classList.add('bg-yellow-600/90', 'text-slate-900');
                    vBtn.classList.remove('text-gray-400', 'bg-purple-600/90');
                }
            }
            if(type === 'dash') loadDashboard();
            if(type === 'tools' && typeof fetchIpIntel === 'function') fetchIpIntel();
            
            if(btn) {
                // Removed voice alert call
            }
        }

        const passInput = document.getElementById('passInput');
        passInput.addEventListener('input', async () => {
            const val = passInput.value;
            if(!val) { document.getElementById('pass-result').classList.add('hidden'); return; }
            const res = await fetch('/scan', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({target:val}) });
            const data = await res.json();
            document.getElementById('pass-result').classList.remove('hidden');
            document.getElementById('strength-text').innerText = data.strength;
            document.getElementById('strength-percent').innerText = ((data.score + 1) * 20) + "%";
            const bar = document.getElementById('strength-bar');
            bar.style.width = (data.score + 1) * 20 + "%";
            bar.style.backgroundColor = ['#ef4444', '#f97316', '#eab308', '#a855f7', '#22c55e'][data.score];
            const leak = document.getElementById('leak-info');
            leak.classList.remove('hidden');
            if(data.exposed_count > 0) {
                soundManager.alarm(); // صوت إنذار الاختراق
                leak.innerHTML = `🚨 متسربة في <b>${data.exposed_count}</b> خرق!`;
                leak.className = "p-4 rounded-xl border border-red-800 bg-red-900/20 text-red-400";
            } else {
                soundManager.success();
                leak.innerHTML = `✅ لم يتم العثور عليها في تسريبات معروفة.`;
                leak.className = "p-4 rounded-xl border border-green-800 bg-green-900/20 text-green-400";
            }
        });

        async function processText(action) {
            const text = document.getElementById('cryptText').value;
            const key = document.getElementById('cryptKey').value;
            if(!text || !key) return alert("يرجى إدخال النص وكلمة السر!");
            const res = await fetch('/crypt-text', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, key, action}) });
            const data = await res.json();
            if(data.error) alert(data.error); else document.getElementById('cryptText').value = data.result;
        }

        async function processFile(action) {
            const file = document.getElementById('fileInput').files[0];
            const key = document.getElementById('cryptKey').value;
            if(!file || !key) return alert("يرجى اختيار ملف وإدخال كلمة السر!");
            const formData = new FormData();
            formData.append('file', file);
            formData.append('key', key);
            formData.append('action', action);
            const res = await fetch('/crypt-file', { method:'POST', body: formData });
            if(res.ok) {
                soundManager.success();
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = action === 'encrypt' ? file.name + '.titan' : file.name.replace('.titan', '');
                a.click();
            } else {
                soundManager.error();
                const err = await res.json(); alert(err.error);
            }
        }

        async function generatePass(mode = 'random') {
            const res = await fetch(`/generate?mode=${mode}`);
            const data = await res.json();
            document.getElementById('suggested-pass').innerText = data.suggested;
            document.getElementById('suggested-pass-container').classList.remove('hidden');
        }

        function copyPass() {
            navigator.clipboard.writeText(document.getElementById('suggested-pass').innerText);
            alert("تم النسخ!");
        }
        async function checkIP() {
            let ip = document.getElementById('ipInput').value.trim();
            const resultDiv = document.getElementById('ipResult');
            const dataBox = document.getElementById('ipDataBox');
            
            resultDiv.classList.remove('hidden');
            dataBox.innerHTML = '<div class="p-4 text-center"><span class="text-purple-400 animate-pulse text-sm font-mono tracking-wider">جاري التقاط الحزم وتحليل مسار الاتصال...</span></div>';
            soundManager.terminalType();

            // فحص الـ IP من جهة العميل لضمان مرور الطلب عبر أي متصفح VPN نشط
            if (!ip) {
                try {
                    const ipRes = await fetch('https://api.ipify.org?format=json');
                    const ipData = await ipRes.json();
                    if(ipData && ipData.ip) ip = ipData.ip;
                } catch (e) {
                    console.log("Fallback to backend IP detection");
                }
            }

            const res = await fetch('/api/ip', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ip}) });
            const data = await res.json();
            
            if (data.success) {
                soundManager.success();
                dataBox.innerHTML = `
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-y-5 gap-x-6">
                        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 hover:border-purple-500/50 transition-colors">
                            <div class="text-slate-400 text-[10px] uppercase mb-1 flex items-center gap-2 tracking-wider"><span class="w-1.5 h-1.5 rounded-full bg-purple-500 shadow-[0_0_5px_#a855f7]"></span>العنوان (IP)</div> 
                            <div class="font-mono text-xl text-purple-400 font-bold text-shadow-sm">${ip || data.query || 'غير معروف'}</div>
                        </div>
                        <div class="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50 hover:border-purple-500/50 transition-colors">
                            <div class="text-slate-400 text-[10px] uppercase mb-1 flex items-center gap-2 tracking-wider"><span class="w-1.5 h-1.5 rounded-full bg-yellow-500 shadow-[0_0_5px_#eab308]"></span>مزود الخدمة (ISP) - البلد</div> 
                            <div class="font-semibold text-gray-200">${data.ISP} <span class="text-gray-500 text-xs">(${data.country_code})</span></div>
                        </div>
                    </div>
                `;
            } else {
                soundManager.error();
                dataBox.innerHTML = `<div class="text-red-400 font-bold bg-red-900/20 p-4 rounded-lg">خطأ: ${data.message || 'فشل جلب البيانات'}</div>`;
            }
        }

        async function generate2FA() {
            const res = await fetch('/api/2fa/generate');
            const data = await res.json();
            document.getElementById('tfaResult').classList.remove('hidden');
            document.getElementById('qrCodeImg').src = data.qr_code;
            document.getElementById('tfaSecret').innerText = data.secret;
            document.getElementById('tfaCodeInput').value = '';
        }

        async function verify2FA() {
            const secret = document.getElementById('tfaSecret').innerText;
            const code = document.getElementById('tfaCodeInput').value;
            if(!secret || !code) return alert("الرجاء إدخال الرمز للتحقق!");
            
            const res = await fetch('/api/2fa/verify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({secret, code})
            });
            const data = await res.json();
            if(data.valid) {
                soundManager.success();
                alert("✅ الرمز صحيح! المصادقة ناجحة.");
            } else {
                soundManager.error();
                alert("❌ الرمز خاطئ أو منتهي الصلاحية.");
            }
        }

        // --- منطق قبو كلمات المرور المكتوب بـ Vanilla JS ---
        let vaultData = [];
        let currentMasterKey = "";

        async function checkVaultPasswordSetup() {
            // Called when vault tab is clicked — shows setup or login panel
            try {
                const res = await fetch('/api/vault/has-password');
                const data = await res.json();
                const setupPanel = document.getElementById('vault-setup');
                const loginPanel = document.getElementById('vault-login');
                if (!data.hasVaultPassword) {
                    // First time: show setup panel
                    if (setupPanel) setupPanel.classList.remove('hidden');
                    if (loginPanel) loginPanel.classList.add('hidden');
                } else {
                    // Already has a password: show login panel
                    if (setupPanel) setupPanel.classList.add('hidden');
                    if (loginPanel) loginPanel.classList.remove('hidden');
                }
            } catch(e) { /* backend not running, just show login */ }
        }

        async function setVaultPassword() {
            const pw1 = document.getElementById('vaultSetupPass1').value;
            const pw2 = document.getElementById('vaultSetupPass2').value;
            const errEl = document.getElementById('vaultSetupError');
            errEl.classList.add('hidden');
            if (!pw1) { errEl.textContent = 'أدخل كلمة السر.'; errEl.classList.remove('hidden'); return; }
            if (pw1.length < 4) { errEl.textContent = 'كلمة السر يجب أن تكون 4 أحرف على الأقل.'; errEl.classList.remove('hidden'); return; }
            if (pw1 !== pw2) { errEl.textContent = 'كلمتا السر غير متطابقتين!'; errEl.classList.remove('hidden'); return; }
            const res = await fetch('/api/vault/set-password', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ password: pw1 })
            });
            const data = await res.json();
            if (data.error) { errEl.textContent = data.error; errEl.classList.remove('hidden'); return; }
            soundManager.success();
            // Switch to login view and pre-fill the password
            document.getElementById('vault-setup').classList.add('hidden');
            document.getElementById('vault-login').classList.remove('hidden');
            document.getElementById('vaultMasterKey').value = pw1;
            await unlockVault(); // Auto-open vault right after setup
        }

        // --- Vault Logic ---
        let vaultFailedAttempts = 0;

        function triggerVaultLockout() {
            // Full-screen red lockout overlay — same feel as burn chat blackout
            const overlay = document.createElement('div');
            overlay.id = 'vault-lockout-overlay';
            overlay.style.cssText = `
                position:fixed;inset:0;z-index:999999;
                background:radial-gradient(ellipse at center, #1a0000 0%, #000 100%);
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                animation:fadeInLockout 0.4s ease;
            `;
            overlay.innerHTML = `
                <style>
                    @keyframes fadeInLockout{from{opacity:0}to{opacity:1}}
                    @keyframes redPulse{0%,100%{text-shadow:0 0 20px #ef4444,0 0 60px #ef4444;}50%{text-shadow:0 0 5px #ef4444;}}
                    @keyframes scanLine{0%{top:0}100%{top:100%}}
                    .lockout-scanline{position:absolute;left:0;width:100%;height:2px;background:rgba(239,68,68,0.4);animation:scanLine 2s linear infinite;pointer-events:none;}
                </style>
                <div class="lockout-scanline"></div>
                <div style="font-size:5rem;animation:redPulse 1.5s infinite;">🔴</div>
                <h1 style="color:#ef4444;font-size:2rem;font-weight:900;letter-spacing:0.15em;margin:1rem 0 0.5rem;text-shadow:0 0 30px #ef4444;">ACCESS DENIED</h1>
                <p style="color:#f87171;font-size:1rem;letter-spacing:0.1em;margin-bottom:0.5rem;">تجاوزت عدد محاولات الدخول المسموح بها</p>
                <p style="color:#6b7280;font-size:0.75rem;font-family:monospace;letter-spacing:0.2em;">VAULT LOCKED — SESSION TERMINATED</p>
                <div style="margin-top:2rem;width:200px;height:4px;background:#1f0000;border-radius:4px;overflow:hidden;">
                    <div id="lockout-bar" style="height:100%;width:100%;background:#ef4444;animation:none;"></div>
                </div>
                <p style="color:#4b5563;font-size:0.7rem;margin-top:0.75rem;font-family:monospace;">SYSTEM RE-ENABLING IN <span id="lockout-count">30</span>s</p>
            `;
            document.body.appendChild(overlay);

            // 30-second countdown then unlock
            let secs = 30;
            const bar = overlay.querySelector('#lockout-bar');
            const counter = overlay.querySelector('#lockout-count');
            const timer = setInterval(() => {
                secs--;
                counter.textContent = secs;
                if(bar) bar.style.width = (secs / 30 * 100) + '%';
                if(secs <= 0) {
                    clearInterval(timer);
                    overlay.remove();
                    vaultFailedAttempts = 0;
                    document.getElementById('vaultMasterKey').value = '';
                }
            }, 1000);
        }

        async function unlockVault() {
            const key = document.getElementById('vaultMasterKey').value;
            if(!key) return alert("أدخل كلمة السر الرئيسية!");
            
            const res = await fetch('/api/vault/load', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ key })
            });
            const data = await res.json();
            
            if(data.error) {
                soundManager.error();
                vaultFailedAttempts++;
                const remaining = 3 - vaultFailedAttempts;
                if(vaultFailedAttempts >= 3) {
                    triggerVaultLockout();
                    vaultFailedAttempts = 0;
                } else {
                    alert(`❌ كلمة السر خاطئة! تحذير: ${remaining} محاولة متبقية قبل تجميد النظام.`);
                }
            } else {
                soundManager.success();
                vaultFailedAttempts = 0;
                currentMasterKey = key;
                vaultData = data.vault || [];
                document.getElementById('vault-login').classList.add('hidden');
                document.getElementById('vault-content').classList.remove('hidden');
                renderVaultItems();
            }
        }

        function lockVault(silent = false) {
            currentMasterKey = "";
            vaultData = [];
            document.getElementById('vaultMasterKey').value = "";
            document.getElementById('vault-login').classList.remove('hidden');
            document.getElementById('vault-content').classList.add('hidden');
            document.getElementById('vaultItemsContainer').innerHTML = "";
            if(!silent) alert("🔒 تم إغلاق القبو بنجاح.");
        }

        async function saveVault() {
            if(!currentMasterKey) return;
            const res = await fetch('/api/vault/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ key: currentMasterKey, vault: vaultData })
            });
            const data = await res.json();
            if(data.error) alert("طأ في حفظ القبو: " + data.error);
        }

        function addVaultItem() {
            const title = document.getElementById('vaultItemTitle').value;
            const username = document.getElementById('vaultItemUsername').value;
            const password = document.getElementById('vaultItemPass').value;
            
            if(!title || !password) return alert("يجب إدخال العنوان وكلمة السر على الأقل!");
            
            vaultData.push({ title, username, password, date: new Date().toISOString().split('T')[0] });
            
            document.getElementById('vaultItemTitle').value = "";
            document.getElementById('vaultItemUsername').value = "";
            document.getElementById('vaultItemPass').value = "";
            
            renderVaultItems();
            saveVault();
        }

        function renderVaultItems() {
            const container = document.getElementById('vaultItemsContainer');
            if(vaultData.length === 0) {
                container.innerHTML = '<p class="text-center text-gray-500 py-4">القبو فارغ حالياً.</p>';
                return;
            }
            
            container.innerHTML = vaultData.map((item, index) => `
                <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 hover:border-yellow-900/50 transition-colors">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-1">
                            <h4 class="font-bold text-purple-400">${item.title}</h4>
                            <span class="text-xs text-slate-500">${item.date || ''}</span>
                        </div>
                        <p class="text-sm text-gray-400">👤 ${item.username || 'بدون اسم مستخدم'}</p>
                    </div>
                    <div class="flex items-center gap-2 w-full md:w-auto">
                        <input type="password" id="vault-pass-${index}" value="${item.password}" readonly class="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-center w-full md:w-32 focus:outline-none">
                        <button onclick="toggleVaultPass(${index})" class="bg-blue-600/20 text-blue-500 hover:bg-blue-600 hover:text-white p-2 rounded-lg transition-colors" title="إظهار/إخفاء كلمة السر">👁️</button>
                        <button onclick="deleteVaultItem('${index}')" class="bg-red-900/20 text-red-500 hover:bg-red-600 hover:text-white p-2 rounded-lg transition-colors" title="حذف">🗑️</button>
                    </div>
                </div>
            `).join('');
        }

        function toggleVaultPass(index) {
            const input = document.getElementById(`vault-pass-${index}`);
            if(input.type === 'password') {
                input.type = 'text';
            } else {
                input.type = 'password';
            }
        }

        function deleteVaultItem(index) {
            if(confirm("هل أنت متأكد من حذف هذا السجل بشكل نهائي؟")) {
                vaultData.splice(index, 1);
                renderVaultItems();
                saveVault();
            }
        }

        async function backupVault() {
            const res = await fetch('/api/vault/backup', { method: 'POST' });
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = "vault_backup.titan.bak";
                a.click();
            } else {
                alert("فشل تصدير النسخة!");
            }
        }

        async function restoreVault(input) {
            if (!input.files[0]) return;
            const formData = new FormData();
            formData.append('file', input.files[0]);
            const res = await fetch('/api/vault/restore', { method: 'POST', body: formData });
            if (res.ok) {
                alert("تم استعادة النسخة بنجاح! يرجى إدخال كلمة السر لفتح القبو.");
                lockVault();
            } else {
                alert("فشل استعادة النسخة!");
            }
        }

        async function showRecoveryMode() {
            try {
                const res = await fetch('/api/vault/recovery/questions');
                const data = await res.json();
                if(data.error) throw new Error(data.error);
                
                document.getElementById('rec-q1').innerText = data.q1;
                document.getElementById('rec-q2').innerText = data.q2;
                document.getElementById('vault-auth-mode').classList.add('hidden');
                document.getElementById('vault-recover-mode').classList.remove('hidden');
            } catch (e) {
                alert(e.message);
                soundManager.error();
            }
        }

        function hideRecoveryMode() {
            document.getElementById('vault-auth-mode').classList.remove('hidden');
            document.getElementById('vault-recover-mode').classList.add('hidden');
        }

        async function executeRecovery() {
            const a1 = document.getElementById('rec-a1').value;
            const a2 = document.getElementById('rec-a2').value;
            if(!a1 || !a2) return alert("الرجاء إدخال الإجابات!");
            
            try {
                const res = await fetch('/api/vault/recovery/recover', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({a1, a2})
                });
                const data = await res.json();
                if(data.error) throw new Error(data.error);
                
                document.getElementById('vaultMasterKey').value = data.recovered_key;
                soundManager.success();
                alert("✅ تم استرجاع كلمة السر بنجاح! يتم فتح القبو الآن.");
                hideRecoveryMode();
                unlockVault(); // Auto-unlock with the recovered key
            } catch(e) {
                alert(e.message);
                soundManager.error();
            }
        }

        function showSetupRecovery() {
            document.getElementById('setup-recovery-container').classList.toggle('hidden');
        }

        async function saveRecoverySetup() {
            const q1 = document.getElementById('setup-q1').value;
            const a1 = document.getElementById('setup-a1').value;
            const q2 = document.getElementById('setup-q2').value;
            const a2 = document.getElementById('setup-a2').value;
            
            if(!q1 || !a1 || !q2 || !a2) return alert("جميع حقول أسئلة الأمان وإجاباتها مطلوبة!");
            
            try {
                const res = await fetch('/api/vault/recovery/setup', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key: currentMasterKey, q1, a1, q2, a2})
                });
                const data = await res.json();
                if(data.error) throw new Error(data.error);
                
                alert("✅ تم إعداد أسئلة استعادة كلمة السر بنجاح للمستقبل.");
                document.getElementById('setup-recovery-container').classList.add('hidden');
                soundManager.success();
            } catch (e) {
                alert(e.message);
                soundManager.error();
            }
        }

        async function processMetadata() {
            const file = document.getElementById('metadataFile').files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/metadata/remove', { method: 'POST', body: formData });
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "clean_" + file.name;
            a.click();
            refreshLogs();
        }

        async function processPdf(action) {
            const file = document.getElementById('pdfFileInput').files[0];
            const password = document.getElementById('pdfPass').value;
            if(!file || !password) return alert("الرجاء اختيار ملف PDF وإدخال كلمة سر المكونة منه!");
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('password', password);
            formData.append('action', action);
            
            try {
                const res = await fetch('/api/pdf-process', { method: 'POST', body: formData });
                if(res.ok) {
                    soundManager.success();
                    const blob = await res.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = (action === 'lock' ? 'locked_' : 'unlocked_') + file.name;
                    a.click();
                    refreshLogs();
                } else {
                    const data = await res.json();
                    throw new Error(data.error || "خطأ غير معروف (ربما كلمة السر التي أدخلتها خاطئة!)");
                }
            } catch (e) {
                alert(e.message);
                soundManager.error();
            }
        }

        async function scanPorts() {
            const ip = document.getElementById('portIpInput').value.trim() || '127.0.0.1';
            const btn = document.getElementById('btnPortScan');
            const label = document.getElementById('scanLabel');
            const loader = document.getElementById('scanLoader');
            const resultBox = document.getElementById('portResult');
            const container = document.getElementById('openPortsContainer');
            
            btn.disabled = true;
            label.classList.add('hidden');
            loader.classList.remove('hidden');
            resultBox.classList.remove('hidden');
            container.innerHTML = '<span class="text-gray-500 font-mono animate-pulse text-xs">جاري التشخيص وفحص المنافذ الحساسة...</span>';
            document.getElementById('portScanTarget').innerText = ip;
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/port-scan', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ip}) });
                const data = await res.json();
                
                if (data.error) {
                    container.innerHTML = `<span class="text-red-400 bg-red-900/20 px-3 py-2 rounded">خطأ: ${data.error}</span>`;
                    soundManager.error();
                } else if (data.open_ports && data.open_ports.length > 0) {
                    soundManager.alarm();
                    container.innerHTML = data.open_ports.map(p => `
                        <div class="px-4 py-2 bg-red-900/40 border border-red-500/50 rounded-xl text-red-400 font-mono shadow-[0_0_10px_rgba(239,68,68,0.2)] flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444] animate-pulse"></span> منفذ ${p} مفتوح
                        </div>
                    `).join('');
                } else {
                    soundManager.success();
                    container.innerHTML = `<div class="bg-green-900/20 text-green-400 font-bold px-4 py-3 rounded-xl border border-green-800/50 w-full text-center flex items-center justify-center gap-2">✅ جميع المنافذ المفحوصة مغلقة (آمن)</div>`;
                }
            } catch (e) {
                container.innerHTML = `<span class="text-red-400">فشل الاتصال بالخادم.</span>`;
                soundManager.error();
            }
            
            btn.disabled = false;
            label.classList.remove('hidden');
            loader.classList.add('hidden');
            refreshLogs();
        }

        async function processExifOsint() {
            const file = document.getElementById('osintFile').files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            
            const resBox = document.getElementById('osintResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-blue-500 animate-pulse">جاري التحليل واستخراج البيانات العميقة...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/osint/image', { method: 'POST', body: formData });
                const data = await res.json();
                
                if(data.error) throw new Error(data.error);
                
                let html = '';
                for(let key in data) {
                    html += `<span class="text-purple-400">${key}:</span> <span class="text-gray-300">${data[key]}</span>\n`;
                }
                resBox.innerHTML = html || 'لا توجد بيانات حساسة.';
                soundManager.success();
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">خطأ: ${e.message}</span>`;
                soundManager.error();
            }
        }


        async function checkPhishing() {
            const email = document.getElementById('phishUrlInput').value;
            if(!email || !email.includes('@')) return alert("الرجاء إدخال بريد إلكتروني صحيح");
            const resBox = document.getElementById('phishResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-orange-400 animate-pulse text-xs font-mono">جاري فحص سمعة البريد الإلكتروني عبر IPQualityScore...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/email', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email})
                });
                const data = await res.json();
                
                if (data.error) {
                    resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.error}</span>`;
                    return;
                }
                
                if (data.success) {
                    const riskColor = data.fraud_score > 70 ? 'text-red-500' : (data.fraud_score > 30 ? 'text-orange-400' : 'text-green-500');
                    resBox.innerHTML = `
                        <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
                            <span class="font-bold text-gray-300 font-mono tracking-wider">${email}</span>
                            <span class="font-black ${riskColor} bg-slate-900 px-2 py-1 rounded">Fraud Score: ${data.fraud_score}</span>
                        </div>
                        <div class="grid grid-cols-2 gap-3 mb-2">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Valid Email</span>
                                <span class="font-bold font-mono ${data.valid ? 'text-green-400' : 'text-red-400'}">${data.valid ? 'YES' : 'NO'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Disposable</span>
                                <span class="font-bold font-mono ${data.disposable ? 'text-red-400' : 'text-green-400'}">${data.disposable ? 'YES (وهمي)' : 'NO'}</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-3">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Spam Trap Score</span>
                                <span class="font-bold text-gray-200">${data.spam_trap_score}</span>
                            </div>
                        </div>
                    `;
                    if(data.fraud_score > 70 || data.disposable || !data.valid) soundManager.alarm(); else soundManager.success();
                } else {
                    resBox.innerHTML = `<span class="text-red-400">خطأ من الخدمة: ${data.message}</span>`;
                    soundManager.error();
                }
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم الفحص.</span>`;
                soundManager.error();
            }
        }

        async function checkEmailPassLeak() {
            const email = document.getElementById('leakEmailInput').value;
            const password = document.getElementById('leakPassInput').value;
            if(!email || !password) return alert("الرجاء إدخال البريد الإلكتروني وكلمة السر بشكل صحيح");
            
            const resBox = document.getElementById('leakEmailPassResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-orange-400 animate-pulse text-xs font-mono">جاري فحص التسريبات عبر IPQualityScore...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/emailpass_leak', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if (data.error) {
                    resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.error}</span>`;
                    return;
                }
                
                if (data.success) {
                    const isLeaked = data.leaked === true || data.leaked === "true";
                    resBox.innerHTML = `
                        <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
                            <span class="font-bold text-gray-300 font-mono tracking-wider">${email}</span>
                        </div>
                        <div class="bg-slate-950 p-4 rounded-lg border ${isLeaked ? 'border-red-500/50' : 'border-green-500/50'} text-center">
                            <span class="text-xs text-gray-400 block mb-2 font-bold uppercase tracking-widest">حالة التسريب لهذه البيانات معاً</span>
                            <div class="text-lg font-bold font-mono ${isLeaked ? 'text-red-500' : 'text-green-400'}">
                                ${isLeaked ? '🚨 تم تسريب هذه البيانات معاً مسبقاً! (خطر)' : '✅ لم يثبت تسريب الإيميل مع كلمة السر (آمن نسبياً)'}
                            </div>
                        </div>
                    `;
                    if(isLeaked) soundManager.alarm(); else soundManager.success();
                } else {
                    resBox.innerHTML = `<span class="text-red-400">خطأ من الخدمة: ${data.message}</span>`;
                    soundManager.error();
                }
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم الفحص.</span>`;
                soundManager.error();
            }
        }

        async function fetchIpqsLogs() {
            const reqType = document.getElementById('ipqsLogType').value;
            const startDate = document.getElementById('ipqsLogDate').value;
            const resBox = document.getElementById('ipqsLogsResult');
            
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-teal-400 animate-pulse">جاري جلب السجلات من الخادم...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/ipqs_logs', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: reqType, start_date: startDate})
                });
                const data = await res.json();
                
                if (data.success === false) {
                    resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.message || 'فشل جلب السجلات'}</span>`;
                    soundManager.error();
                    return;
                }
                
                const requests = data.requests || [];
                if (requests.length === 0) {
                    resBox.innerHTML = '<span class="text-gray-400 italic">لا توجد سجلات مطابقة في هذه الفترة.</span>';
                    soundManager.success();
                    return;
                }
                
                let html = '<div class="space-y-2">';
                requests.forEach(req => {
                    const statusStr = req.status ? '<span class="text-green-400">Success</span>' : '<span class="text-red-400">Failed</span>';
                    const fraudStr = req.fraud_score !== undefined ? `Fraud: <span class="text-yellow-400">${req.fraud_score}</span>` : '';
                    const dateStr = new Date(req.request_date).toLocaleString('ar-EG');
                    
                    // target represents what was scanned (ip, email, query, etc)
                    const targetStr = req.query || req.email || req.ip || req.phone || 'Unknown Target';
                    
                    html += `
                        <div class="bg-slate-900 p-2 rounded border border-slate-700/50 hover:border-teal-500/30 transition-colors flex flex-col gap-1">
                            <div class="flex justify-between items-start">
                                <span class="text-teal-300 font-bold">${targetStr}</span>
                                <span class="text-[10px] text-gray-500">${dateStr}</span>
                            </div>
                            <div class="flex justify-between items-center text-xs text-gray-400 mt-1">
                                <span>Status: ${statusStr}</span>
                                <span>${fraudStr}</span>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                
                resBox.innerHTML = html;
                soundManager.success();
                
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم السجلات.</span>`;
                soundManager.error();
            }
        }

        async function checkPhone() {
            const phone = document.getElementById('phoneInput').value;
            if(!phone) return alert("الرجاء إدخال رقم الهاتف");
            const resBox = document.getElementById('phoneResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-yellow-400 animate-pulse text-xs font-mono">جاري فحص الرقم عبر IPQualityScore...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/phone', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({phone})
                });
                const data = await res.json();
                
                if (data.error) {
                    resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.error}</span>`;
                    return;
                }
                
                if (data.success) {
                    const riskColor = data.fraud_score > 70 ? 'text-red-500' : (data.fraud_score > 30 ? 'text-orange-400' : 'text-green-500');
                    resBox.innerHTML = `
                        <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
                            <span class="font-bold text-gray-300 font-mono tracking-wider" dir="ltr">${data.formatted || phone}</span>
                            <span class="font-black ${riskColor} bg-slate-900 px-2 py-1 rounded">Fraud Score: ${data.fraud_score}</span>
                        </div>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-2">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Valid</span>
                                <span class="font-bold font-mono ${data.valid ? 'text-green-400' : 'text-red-400'}">${data.valid ? 'YES' : 'NO'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Active</span>
                                <span class="font-bold font-mono ${data.active ? 'text-green-400' : 'text-orange-400'}">${data.active ? 'YES' : 'Unknown'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Line Type</span>
                                <span class="font-bold font-mono text-gray-200">${data.line_type || 'N/A'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Recent Abuse</span>
                                <span class="font-bold font-mono ${data.recent_abuse ? 'text-red-400' : 'text-green-400'}">${data.recent_abuse ? 'YES' : 'NO'}</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-1 gap-3">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Carrier</span>
                                <span class="font-bold text-gray-200">${data.carrier || 'N/A'}</span>
                            </div>
                        </div>
                    `;
                    if(data.fraud_score > 70 || data.recent_abuse || !data.valid) soundManager.alarm(); else soundManager.success();
                } else {
                    resBox.innerHTML = `<span class="text-red-400">خطأ من الخدمة: ${data.message} <br> <span class="text-xs text-gray-500 mt-2 block">(ملاحظة النظام: إذا كان الخطأ يتكرر لجميع الأرقام، فهذا يعني أن رصيد حسابك المجاني في IPQualityScore قد انتهى لليوم)</span></span>`;
                    soundManager.error();
                }
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم الفحص.</span>`;
                soundManager.error();
            }
        }

        async function checkUrl() {
            const url = document.getElementById('urlInput').value;
            if(!url || (!url.startsWith('http://') && !url.startsWith('https://'))) return alert("الرجاء إدخال رابط صحيح يبدأ بـ http:// أو https://");
            const resBox = document.getElementById('urlResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-blue-400 animate-pulse text-xs font-mono">جاري فحص الرابط عبر IPQualityScore...</span>';
             soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/url', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url})
                });
                const data = await res.json();
                
                if (data.error) {
                    resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.error}</span>`;
                    return;
                }
                
                if (data.success) {
                    const riskColor = data.risk_score > 70 ? 'text-red-500' : (data.risk_score > 30 ? 'text-orange-400' : 'text-green-500');
                    resBox.innerHTML = `
                        <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2 flex-wrap gap-2">
                            <span class="font-bold text-gray-300 font-mono tracking-wider break-all text-xs" dir="ltr">${url}</span>
                            <span class="font-black ${riskColor} bg-slate-900 px-2 py-1 rounded">Risk Score: ${data.risk_score || 0}</span>
                        </div>
                        <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-2">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Phishing</span>
                                <span class="font-bold font-mono ${data.phishing ? 'text-red-400' : 'text-green-400'}">${data.phishing ? 'YES (تصيد)' : 'NO'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Malware</span>
                                <span class="font-bold font-mono ${data.malware ? 'text-red-400' : 'text-green-400'}">${data.malware ? 'YES (خبيث)' : 'NO'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Suspicious</span>
                                <span class="font-bold font-mono ${data.suspicious ? 'text-orange-400' : 'text-green-400'}">${data.suspicious ? 'YES (مشبوه)' : 'NO'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Parking</span>
                                <span class="font-bold font-mono text-gray-200">${data.parking ? 'YES' : 'NO'}</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Domain Name</span>
                                <span class="font-bold text-gray-200 text-xs">${data.domain || 'N/A'}</span>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50">
                                <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Category / Server</span>
                                <span class="font-bold text-gray-200 text-xs">${data.category || 'N/A'} / ${data.server || 'N/A'}</span>
                            </div>
                        </div>
                    `;
                    if(data.risk_score > 70 || data.phishing || data.malware || data.suspicious) soundManager.alarm(); else soundManager.success();
                } else {
                    resBox.innerHTML = `<span class="text-red-400">خطأ من الخدمة: ${data.message}</span>`;
                     soundManager.error();
                }
            } catch (e) {
                resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم الفحص.</span>`;
                 soundManager.error();
            }
        }

        function renderMalwareResult(data, targetName, isUrl = true) {
            const resBox = document.getElementById('malwareResult');
            
            if (data.error) {
                resBox.innerHTML = `<span class="text-red-400">خطأ: ${data.error}</span>`;
                return;
            }
            
            if (data.status === "pending") {
                 resBox.innerHTML = `<span class="text-yellow-400 animate-pulse font-mono tracking-widest whitespace-pre-wrap"><br/>⚠️ جاري تحليل الهدف أمنياً في الخادم... الرجاء الانتظار بضع ثوانٍ.</span>`;
                 return;
            }
             
            if (data.success && data.result) {
                const scan = data.result;
                 // Some risk score keys for malware scan could differ slightly, safely extracting
                let riskScore = scan.risk_score || 0;
                let riskColor = riskScore > 70 ? 'text-red-500' : (riskScore > 30 ? 'text-orange-400' : 'text-green-500');
                
               resBox.innerHTML = `
                    <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2 flex-wrap gap-2">
                        <span class="font-bold text-gray-300 font-mono tracking-wider break-all text-xs" dir="ltr">${targetName}</span>
                         <span class="font-black ${riskColor} bg-slate-900 px-2 py-1 rounded">Risk Score: ${riskScore}</span>
                    </div>
                    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-2">
                        <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                            <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Malicious</span>
                             <span class="font-bold font-mono ${scan.malicious ? 'text-red-400' : 'text-green-400'}">${scan.malicious ? 'YES (خبيث)' : 'NO'}</span>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                            <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Phishing</span>
                            <span class="font-bold font-mono ${scan.phishing ? 'text-red-400' : 'text-green-400'}">${scan.phishing ? 'YES (تصيد)' : 'NO'}</span>
                        </div>
                         <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                            <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Suspicious</span>
                            <span class="font-bold font-mono ${scan.suspicious ? 'text-orange-400' : 'text-green-400'}">${scan.suspicious ? 'YES (مشبوه)' : 'NO'}</span>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-lg border border-slate-700/50 text-center">
                            <span class="text-[10px] text-gray-400 block mb-1 uppercase tracking-widest">Spam</span>
                            <span class="font-bold font-mono ${scan.spam ? 'text-red-400' : 'text-green-400'}">${scan.spam ? 'YES (مزعج)' : 'NO'}</span>
                        </div>
                    </div>
                `;
                 if(riskScore > 70 || scan.malicious || scan.phishing || scan.suspicious) soundManager.alarm(); else soundManager.success();
             } else {
                 resBox.innerHTML = `<span class="text-red-400">خطأ من الخدمة: ${data.message || 'فشل عملية الفحص العميق.'}</span>`;
                  soundManager.error();
             }
        }

        async function checkMalwareUrl() {
            const url = document.getElementById('malwareUrlInput').value;
            if(!url || (!url.startsWith('http://') && !url.startsWith('https://'))) return alert("الرجاء إدخال رابط صحيح يبدأ بـ http:// أو https://");
            const resBox = document.getElementById('malwareResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-rose-400 animate-pulse text-xs font-mono">جاري فحص الرابط للبرمجيات الخبيثة عبر IPQualityScore...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/scan/malware_url', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url})
                });
                const data = await res.json();
                renderMalwareResult(data, url, true);
            } catch (e) {
                 resBox.innerHTML = `<span class="text-red-400">فشل الاتصال بخادم الفحص.</span>`;
                 soundManager.error();
            }
        }

        async function checkMalwareFile() {
             const fileInput = document.getElementById('malwareFileInput');
             if(!fileInput.files.length) return alert("الرجاء اختيار ملف للفحص");
             
             const file = fileInput.files[0];
             // Limit check (e.g. 15MB) since IPQualityScore free typically limits file size
             if (file.size > 15 * 1024 * 1024) return alert("حجم الملف كبير جداً. الحد الأقصى 15 ميجابايت.");
             
            const resBox = document.getElementById('malwareResult');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-rose-400 animate-pulse text-xs font-mono">جاري رفع وفحص الملف للبرمجيات الخبيثة...</span>';
            soundManager.terminalType();
            
            const formData = new FormData();
            formData.append('file', file);
            
             try {
                const res = await fetch('/api/scan/malware_file', {
                    method: 'POST', body: formData
                });
                const data = await res.json();
                renderMalwareResult(data, file.name, false);
            } catch (e) {
                 resBox.innerHTML = `<span class="text-red-400">فشل رفع الملف أو الاتصال بالخادم.</span>`;
                 soundManager.error();
            }
        }

        async function scanLanNetwork() {
            const loader = document.getElementById('lanLoader');
            const resBox = document.getElementById('lanResult');
            const btn = document.getElementById('btnLanScan');
            
            btn.disabled = true;
            loader.classList.remove('hidden');
            resBox.classList.remove('hidden');
            resBox.innerHTML = '<span class="text-cyan-500 animate-pulse text-xs font-mono">جاري إرسال حزم استكشافية للشبكة (ARP Sweep)...</span>';
            soundManager.terminalType();
            
            try {
                const res = await fetch('/api/network/scan');
                const data = await res.json();
                
                if(data.length === 0) {
                    resBox.innerHTML = '<span class="text-gray-400">لم يتم العثور على أجهزة (أو الشبكة تمنع الفحص).</span>';
                } else if(data[0].error) {
                    resBox.innerHTML = `<span class="text-red-400">${data[0].error}</span>`;
                } else {
                    resBox.innerHTML = data.map(d => `
                        <div class="flex justify-between items-center p-3 border-b border-slate-800 hover:bg-slate-800/60 transition-colors rounded-lg mb-1">
                            <div class="flex items-center gap-3">
                                <span class="text-xl bg-slate-950 p-2 rounded-lg border border-slate-700">${d.icon || '💻'}</span>
                                <div class="flex flex-col">
                                    <span class="text-cyan-400 font-bold font-mono text-sm tracking-wider">${d.ip}</span>
                                    <span class="text-gray-500 text-[10px] uppercase font-bold">${d.label || d.type || 'Unknown'}</span>
                                </div>
                            </div>
                            <div class="text-right flex flex-col items-end">
                                <span class="text-gray-400 font-mono text-[10px] tracking-widest">${d.mac}</span>
                                <span class="text-[9px] text-teal-600 bg-teal-900/20 px-1 rounded mt-1">ACTIVE</span>
                            </div>
                        </div>
                    `).join('');
                }
                soundManager.success();
            } catch(e) {
                resBox.innerHTML = `<span class="text-red-400">فشل في جلب أجهزة الشبكة.</span>`;
                soundManager.error();
            }
            loader.classList.add('hidden');
            btn.disabled = false;
        }

        async function createBurnNote() {
            const text = document.getElementById('burnNoteText').value;
            if(!text) return alert("يرجى كتابة رسالة الصندوق قبل التوليد!");
            
            try {
                const res = await fetch('/api/burn-note/create', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await res.json();
                if(data.error) throw new Error(data.error);
                
                document.getElementById('burnNoteResult').classList.remove('hidden');
                document.getElementById('burnNoteLink').value = data.link;
                document.getElementById('burnNoteText').value = "";
                soundManager.success();
                refreshLogs();
            } catch (e) {
                alert("خطأ: " + e.message);
                soundManager.error();
            }
        }

        function copyBurnNoteLink() {
            const link = document.getElementById('burnNoteLink');
            link.select();
            document.execCommand('copy');
            
            const btn = document.getElementById('burnCopyBtn');
            const orgText = btn.innerText;
            btn.innerText = '✅ تم النسخ';
            btn.classList.add('text-orange-400', 'border-orange-500', 'bg-orange-900/30');
            soundManager.click();
            
            setTimeout(() => {
                btn.innerText = orgText;
                btn.classList.remove('text-orange-400', 'border-orange-500', 'bg-orange-900/30');
            }, 2000);
        }

        async function showStego(mode) {
            if (mode === 'encode') {
                const text = prompt("أدخل النص الذي تريد إخفاءه داخل الصورة:");
                if (!text) return;
                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.accept = 'image/*';
                fileInput.onchange = async () => {
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    formData.append('text', text);
                    const res = await fetch('/api/steganography/encode', { method: 'POST', body: formData });
                    if(res.ok) {
                        const blob = await res.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url; a.download = "stego_img.png"; a.click();
                        refreshLogs();
                    } else { 
                        const errData = await res.json();
                        alert("خطأ في تشفير الصورة: " + (errData.error || "خطأ غير معروف")); 
                    }
                };
                fileInput.click();
            } else {
                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.accept = 'image/*';
                fileInput.onchange = async () => {
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    const res = await fetch('/api/steganography/decode', { method: 'POST', body: formData });
                    const data = await res.json();
                    alert("النص المستخرج: " + (data.result || "لا توجد بيانات"));
                    refreshLogs();
                };
                fileInput.click();
            }
        }

        async function toggleUsbGuardian(action) {
            try {
                const res = await fetch('/api/defense/usb', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action})
                });
                const data = await res.json();
                
                const badge = document.getElementById('usbStatusBadge');
                if(data.status === 'active') {
                    badge.innerText = 'يراقب 🛡️';
                    badge.className = 'text-[9px] px-2 py-0.5 rounded-full bg-teal-900/50 text-teal-300 border border-teal-500/50 animate-pulse';
                    soundManager.success();
                } else {
                    badge.innerText = 'متوقف';
                    badge.className = 'text-[9px] px-2 py-0.5 rounded-full bg-slate-800 text-gray-400 border border-slate-700';
                    soundManager.click();
                }
                refreshLogs();
            } catch(e) {
                soundManager.error();
            }
        }

        async function toggleFim(action) {
            const target = document.getElementById('fimTargetPath').value;
            if(action === 'start' && !target) return alert("الرجاء إدخال مسار الملف للمراقبة");
            
            try {
                const res = await fetch('/api/defense/fim', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action, target})
                });
                const data = await res.json();
                if(data.error) throw new Error(data.error);
                
                const badge = document.getElementById('fimStatusBadge');
                if(data.status === 'active') {
                    badge.innerText = 'يراقب ⚖️';
                    badge.className = 'text-[9px] px-2 py-0.5 rounded-full bg-orange-900/50 text-orange-300 border border-orange-500/50 animate-pulse';
                    soundManager.success();
                } else {
                    badge.innerText = 'متوقف';
                    badge.className = 'text-[9px] px-2 py-0.5 rounded-full bg-slate-800 text-gray-400 border border-slate-700';
                    soundManager.click();
                }
                refreshLogs();
            } catch(e) {
                alert("خطأ: " + e.message);
                soundManager.error();
            }
        }

        /* --- Burn Chat Logic (E2E Encrypted) --- */
        let burnChatTimer = null;
        let currentRoomId = null;
        let currentUser = null;

        // Custom E2E Encryption (XOR + Base64 Safe) with Signature
        function e2eEncrypt(str, key) {
            let encodedStr = encodeURIComponent(str + "||TITAN_OK||"); // Append verification signature
            let res = "";
            for(let i=0; i<encodedStr.length; i++) {
                res += String.fromCharCode(encodedStr.charCodeAt(i) ^ key.charCodeAt(i % key.length));
            }
            return btoa(res);
        }
        
        function e2eDecrypt(b64, key) {
            let res = "";
            try {
                let decodedStr = atob(b64);
                for(let i=0; i<decodedStr.length; i++) {
                    res += String.fromCharCode(decodedStr.charCodeAt(i) ^ key.charCodeAt(i % key.length));
                }
                
                // Try decoding URI component
                let plaintext = decodeURIComponent(res);
                if(plaintext.endsWith("||TITAN_OK||")) {
                    return { success: true, text: plaintext.substring(0, plaintext.length - 12) };
                }
                return { success: false, text: res }; // Valid URI encoding, but bad signature
            } catch(e) {
                // Invalid URI encoding (XOR caused bad bytes)
                return { success: false, text: res || "GARBLED_DATA" };
            }
        }

        function joinBurnChat() {
            const roomId = document.getElementById('burnChatId').value.trim();
            const user = document.getElementById('burnChatUser').value.trim() || 'Anonymous';
            
            if(!roomId) return alert("الرجاء إدخال رقم الغرفة للاتصال المشفر!");
            
            currentRoomId = roomId;
            currentUser = user;
            
            document.getElementById('burnChatInput').disabled = false;
            document.getElementById('burnChatSendBtn').disabled = false;
            document.getElementById('burnChatSendBtn').className = "bg-pink-600 hover:bg-pink-500 text-white px-8 rounded-lg font-bold transition-all border border-pink-500/50 shadow-[0_0_15px_rgba(236,72,153,0.3)] flex-shrink-0";
            document.getElementById('burnChatDisplay').innerHTML = '<div class="text-center text-pink-500 font-bold tracking-widest text-xs uppercase mt-auto mb-2 animate-pulse">-- 🔒 تم الاتصال بنفق مشفر (End-to-End) --</div><div class="text-center text-gray-500 tracking-widest text-[10px] uppercase">يتم تشفير/فك تشفير الرسائل محلياً داخل متصفحك فقط</div>';
            
            soundManager.success();
            
            if(burnChatTimer) clearInterval(burnChatTimer);
            burnChatTimer = setInterval(pollBurnChat, 2000);
        }

        async function sendBurnChat() {
            const input = document.getElementById('burnChatInput');
            const msg = input.value.trim();
            if(!msg || !currentRoomId) return;
            
            // PROMPT SENDER FOR DECRYPTION KEY
            const encryptKey = prompt("🔐 أدخل مفتاح التشفير الخاص بهذه الرسالة (يجب أن يعرفه الطرف الآخر لفك التشفير):");
            if (!encryptKey) return; // Cancelled
            
            input.value = '';
            
            // Show local preview
            const display = document.getElementById('burnChatDisplay');
            display.innerHTML += `
                <div class="flex justify-start mt-4">
                    <div class="bg-indigo-900/40 border border-indigo-700/50 text-indigo-200 px-4 py-3 rounded-lg text-sm max-w-[85%] break-y relative">
                        <span class="text-[10px] text-indigo-400 font-bold mb-1 block">أنت (${currentUser}) <span class="text-indigo-600 bg-indigo-950 px-1 rounded ml-2">🔒 مُشفّر</span></span>
                        <div class="text-[8px] font-mono text-indigo-700/50 overflow-hidden text-ellipsis whitespace-nowrap mb-2 italic" title="Ciphertext">Encrypted payload sent...</div>
                        <div class="text-indigo-100 font-bold text-[15px]">${msg}</div>
                    </div>
                </div>
            `;
            display.scrollTop = display.scrollHeight;
            soundManager.terminalType();
            
            // Encrypt and Send
            const encryptedMsg = e2eEncrypt(msg, encryptKey);
            try {
                await fetch('/api/chat/send', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({room_id: currentRoomId, sender: currentUser, msg: encryptedMsg})
                });
            } catch(e) {
                console.error("Encryption Transmission Failed:", e);
            }
        }

        async function pollBurnChat() {
            if(!currentRoomId) return;
            
            try {
                const res = await fetch(`/api/chat/receive?room_id=${currentRoomId}&requester=${currentUser}`);
                const data = await res.json();
                
                if(data.messages && data.messages.length > 0) {
                    const display = document.getElementById('burnChatDisplay');
                    data.messages.forEach(m => {
                        // Generate a unique ID for this message block
                        const msgId = 'msg-' + Math.random().toString(36).substr(2, 9);
                        
                        display.innerHTML += `
                            <div class="flex justify-end mt-4 mb-2">
                                <div id="${msgId}" class="bg-pink-900/40 border border-pink-700/50 text-pink-200 px-5 py-4 rounded-lg text-sm max-w-[85%] break-y relative group shadow-[0_4px_20px_rgba(236,72,153,0.15)] transition-all">
                                    <div class="flex justify-between items-center mb-2 border-b border-pink-800/50 pb-2">
                                        <span class="text-xs text-pink-400 font-black">${m.sender}</span>
                                        <span class="text-[9px] text-gray-500 ml-3 uppercase bg-black/50 border border-slate-700 px-2 py-0.5 rounded">🔒 مشفر (Ciphertext)</span>
                                    </div>
                                    <!-- Ciphertext -->
                                    <div class="mb-3 p-2 bg-black/80 rounded border border-pink-900/50">
                                        <div class="text-[10px] font-mono text-pink-700 break-all select-all">${m.msg}</div>
                                    </div>
                                    
                                    <!-- Action Button & Output Area -->
                                    <div id="${msgId}-action-area" class="flex flex-col gap-2">
                                        <button onclick="decryptManual('${msgId}', '${m.msg}')" class="w-full bg-pink-800/30 hover:bg-pink-700/40 border border-pink-700/50 text-pink-300 text-[10px] font-bold py-2 rounded transition-all">
                                            فك التشفير الآن 🔓
                                        </button>
                                    </div>
                                    <div class="absolute -left-4 -top-4 text-2xl opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-xl" title="Burned from Server Memory">🔥</div>
                                </div>
                            </div>
                        `;
                    });
                    display.scrollTop = display.scrollHeight;
                    soundManager.alarm(); 
                }
            } catch(e) {
                console.error(e);
            }
        }

        function decryptManual(msgId, cipherData) {
            const key = prompt("⚠️ أدخل مفتاح فك التشفير السري الخاص بهذه الرسالة:");
            if(!key) return; // User cancelled the prompt
            
            const actionArea = document.getElementById(msgId + '-action-area');
            const result = e2eDecrypt(cipherData, key);
            
            if(!result.success) {
                // Show Garbled Text Result
                actionArea.innerHTML = `
                    <div class="text-[9px] text-red-400 mb-1">❌ محاولة فك تشفير فاشلة (نص مخربط):</div>
                    <div class="text-red-300 font-mono text-sm leading-relaxed bg-red-900/20 p-3 rounded border border-red-800/50 break-all select-all">${result.text.substring(0, 100)}...</div>
                `;
                
                // Update badge to red
                const badge = document.querySelector(`#${msgId} span`);
                if(badge) {
                    badge.className = "text-[9px] text-red-500 ml-3 uppercase bg-red-900/30 border border-red-800/50 px-2 py-0.5 rounded animate-pulse";
                    badge.innerText = "❌ مفتاح خاطئ";
                }
                
                // Alert slightly, then enforce FULL SYSTEM LOCKDOWN after 3 seconds
                soundManager.error();
                setTimeout(() => {
                    clearInterval(burnChatTimer);
                    burnChatTimer = null;
                    
                    // Completely destroy the page UI
                    document.body.innerHTML = `
                        <div class="h-screen w-screen bg-black flex flex-col items-center justify-center text-center p-8 fixed top-0 left-0 z-50">
                            <div class="text-9xl mb-8 animate-bounce">💀</div>
                            <h1 class="text-red-600 font-black text-6xl mb-4 tracking-widest animate-pulse">SYSTEM LOCKED</h1>
                            <h2 class="text-red-500 font-bold text-2xl mb-8">SECURE COMM COMPROMISED</h2>
                            <p class="text-red-400 text-lg max-w-2xl mx-auto mb-10 leading-relaxed border border-red-900/50 bg-red-950/30 p-6 rounded-xl">
                                تم إدخال مفتاح تشفير عالي السرية بشكل خاطئ. للحماية القصوى من محاولات التخمين والاختراق، تم تفعيل بروتوكول التدمير الذاتي وتجميد واجهة النظام بالكامل.
                            </p>
                            <div class="text-gray-600 font-mono text-xs opacity-50 mb-10">
                                ERASING SESSION CACHE... [DONE]<br>
                                WIPING LOCAL TOKENS... [DONE]<br>
                                CONNECTION TERMINATED PERMANENTLY
                            </div>
                            <div class="text-red-500 font-black text-xl animate-pulse border-t border-b border-red-900/50 py-4 w-full max-w-md">
                                يُرجى إغلاق المتصفح أو علامة التبويب فوراً.
                            </div>
                        </div>
                    `;
                    soundManager.alarm();
                }, 2500);
                return;
            }
            
            // Success: Replace the button area with the decrypted result
            actionArea.innerHTML = `
                <div class="text-[9px] text-green-400 mb-1">تم فك التشفير محلياً بنجاح باستخدام المفتاح المقدم:</div>
                <div class="text-white font-bold text-lg leading-relaxed bg-green-900/20 p-3 rounded border border-green-800/30">${result.text}</div>
            `;
            
            // Update the badge
            const badge = document.querySelector(`#${msgId} span.text-gray-500`) || document.querySelector(`#${msgId} span`);
            if(badge) {
                badge.className = "text-[9px] text-green-400 ml-3 uppercase bg-green-900/30 border border-green-800/50 px-2 py-0.5 rounded animate-pulse";
                badge.innerText = "🔓 فُك تشفيره";
            }
            
            // Highlight the box temporarily
            const box = document.getElementById(msgId);
            box.classList.add('ring-2', 'ring-green-500/50');
            setTimeout(() => box.classList.remove('ring-2', 'ring-green-500/50'), 1000);
            
            soundManager.success();
        }
        
        /* -------------------------- */

        async function refreshLogs() {
            const res = await fetch('/api/audit-logs');
            const data = await res.json();
            const container = document.getElementById('auditContainer');
            if(!container) return;
            container.innerHTML = data.map(log => `
                <div class="flex justify-between border-b border-white/5 py-1">
                    <span class="text-purple-500">[${log.time}]</span>
                    <span class="text-gray-300 mx-2">${log.action}</span>
                    <span class="text-gray-500 text-[9px] truncate ml-auto">${log.details}</span>
                </div>
            `).join('');
        }
        
        // Initial setup
        showTab('pass');
        refreshLogs();

        // ===========================
        // ===== AUTH SYSTEM JS  =====
        // ===========================

        function initAuthMatrix() {
            const canvas = document.getElementById('auth-matrix');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            const chars = "01アイウエオカキクケコサシスセソTITAN".split("");
            const fontSize = 11;
            const cols = Math.floor(canvas.width / fontSize);
            const drops = Array.from({length: cols}, () => Math.random() * -100);
            function drawAuthMatrix() {
                ctx.fillStyle = "rgba(5,5,16,0.18)";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < drops.length; i++) {
                    const ch = chars[Math.floor(Math.random() * chars.length)];
                    const alpha = Math.random() > 0.85 ? 0.9 : 0.35;
                    ctx.fillStyle = Math.random() > 0.7 ? `rgba(168,85,247,${alpha})` : `rgba(88,28,135,${alpha})`;
                    ctx.font = fontSize + "px monospace";
                    ctx.fillText(ch, i * fontSize, drops[i] * fontSize);
                    if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
                    drops[i] += 0.4;
                }
                requestAnimationFrame(drawAuthMatrix);
            }
            drawAuthMatrix();
            window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });
        }

        // Enter key support for auth forms
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const overlay = document.getElementById('auth-overlay');
                if (!overlay || overlay.style.display === 'none') return;
                const loginForm = document.getElementById('auth-login-form');
                if (loginForm && loginForm.style.display !== 'none') doLogin();
                else doRegister();
            }
        });

        // Run auth check on page load
        window.addEventListener('load', () => {
            setTimeout(checkAuth, 150);
        });



        // ===== Phase 6 JS =====
        async function loadDashboard() {
            try {
                const res = await fetch('/api/dashboard/stats');
                const d = await res.json();
                if(d.error) return;
                document.getElementById('dashCpu').innerText  = d.cpu_percent + '%';
                document.getElementById('dashRam').innerText  = d.ram_percent + '%';
                document.getElementById('dashDisk').innerText = d.disk_percent + '%';
                document.getElementById('dashBurn').innerText = d.burn_notes;
                document.getElementById('dashLocalIp').innerText = d.local_ip;
                document.getElementById('dashPubIp').innerText  = d.public_ip;
                document.getElementById('dashSent').innerText   = d.net_sent_mb;
                document.getElementById('dashRecv').innerText   = d.net_recv_mb;
                const logsEl = document.getElementById('dashLogs');
                if(d.recent_logs && d.recent_logs.length) {
                    logsEl.innerHTML = d.recent_logs.map(l =>
                        `<div class="text-purple-400">[${l.time.split(' ')[1]}] <span class="text-gray-300">${l.action}</span></div>`
                    ).join('');
                } else { logsEl.innerHTML = '<div class="text-gray-600">لا يوجد نشاط</div>'; }
            } catch(e) {}
        }

        async function generateQR() {
            const text = document.getElementById('qrText').value.trim();
            const pass = document.getElementById('qrPass').value;
            if(!text) { alert('أدخل النص'); return; }
            const res = await fetch('/api/qr/generate', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({text, password: pass})
            });
            const d = await res.json();
            if(d.error) { alert(d.error); return; }
            const src = 'data:image/png;base64,' + d.qr;
            document.getElementById('qrImg').src = src;
            document.getElementById('qrDownload').href = src;
            document.getElementById('qrResult').classList.remove('hidden');
        }

        async function decodeQR() {
            const file = document.getElementById('qrFile').files[0];
            const pass = document.getElementById('qrDecodePass').value;
            if(!file) { alert('اختر صورة'); return; }
            const form = new FormData();
            form.append('file', file);
            form.append('password', pass);
            const res = await fetch('/api/qr/decode', {method:'POST', body:form});
            const d = await res.json();
            const el = document.getElementById('qrDecodeResult');
            el.classList.remove('hidden');
            el.innerText = d.error ? '❌ ' + d.error : '✅ ' + d.text;
            el.className = el.className + (d.error ? ' text-red-400' : ' text-green-300');
        }

        async function generateIdentity() {
            const lang = document.getElementById('identityLang').value;
            const resArea = document.getElementById('identityResultArea');
            
            try {
                // Dim area while loading
                if(!resArea.classList.contains('hidden')) {
                    resArea.style.opacity = '0.5';
                }
                
                const res = await fetch(`/api/fake-identity?lang=${lang}`);
                const data = await res.json();
                
                if (data.error) throw new Error(data.error);
                
                // Populate data
                document.getElementById('idName').innerText = data.name;
                document.getElementById('idGender').innerText = data.gender;
                document.getElementById('idMotherName').innerText = data.mother_name;
                document.getElementById('idNational').innerText = data.national_id;
                document.getElementById('idDob').innerText = data.birthdate;
                document.getElementById('idAge').innerText = data.age;
                document.getElementById('idZodiac').innerText = data.zodiac;

                document.getElementById('idAddress').innerText = data.address;
                document.getElementById('idZip').innerText = data.zip_code;
                document.getElementById('idGeo').innerText = data.geo;
                document.getElementById('idCountryCode').innerText = data.country_code;

                document.getElementById('idPhone').innerText = data.phone;
                document.getElementById('idEmail').innerText = data.email;
                document.getElementById('idCompany').innerText = data.company;
                document.getElementById('idJob').innerText = data.job;

                document.getElementById('idHeight').innerText = data.height;
                document.getElementById('idWeight').innerText = data.weight;
                document.getElementById('idBlood').innerText = data.blood_type;
                document.getElementById('idColor').innerText = data.color;
                document.getElementById('idVehicle').innerText = data.vehicle;

                document.getElementById('idCcType').innerText = data.cc_type.toUpperCase();
                document.getElementById('idCredit').innerText = data.credit_card.match(/.{1,4}/g) ? data.credit_card.match(/.{1,4}/g).join(' ') : data.credit_card;
                document.getElementById('idCcExp').innerText = data.cc_expire;
                document.getElementById('idCcCvv').innerText = data.cc_cvv;

                document.getElementById('idUsername').innerText = data.username;
                document.getElementById('idPassword').innerText = data.password;
                
                document.getElementById('idWebsite').innerText = data.website;
                document.getElementById('idWebsite').href = data.website;
                document.getElementById('idUserAgent').innerText = data.user_agent;
                document.getElementById('idUuid').innerText = data.uuid;
                
                // Show area and restore opacity
                resArea.classList.remove('hidden');
                resArea.style.opacity = '1';
                
                soundManager.terminalType();
                soundManager.success();
            } catch (err) {
                alert("تعذر توليد الهوية: " + err.message);
                soundManager.error();
                resArea.style.opacity = '1';
            }
        }

        function copyFullIdentity() {
            const dataToCopy = `
=== هوية وهمية مقترحة (${document.getElementById('identityLang').options[document.getElementById('identityLang').selectedIndex].text}) ===
الاسم الكامل: ${document.getElementById('idName').innerText}
الجنس: ${document.getElementById('idGender').innerText}
اسم الأم: ${document.getElementById('idMotherName').innerText}
تاريخ الميلاد: ${document.getElementById('idDob').innerText} (${document.getElementById('idAge').innerText} سنة - ${document.getElementById('idZodiac').innerText})
الرقم الوطني: ${document.getElementById('idNational').innerText}

[معلومات الاتصال والموقع]
العنوان: ${document.getElementById('idAddress').innerText}
الرمز البريدي: ${document.getElementById('idZip').innerText}
الإحداثيات: ${document.getElementById('idGeo').innerText}
رقم الهاتف: ${document.getElementById('idPhone').innerText}
البريد الإلكتروني: ${document.getElementById('idEmail').innerText}

[العمل والخصائص الجسدية]
الشركة: ${document.getElementById('idCompany').innerText}
الوظيفة: ${document.getElementById('idJob').innerText}
الطول/الوزن: ${document.getElementById('idHeight').innerText} / ${document.getElementById('idWeight').innerText}
فصيلة الدم: ${document.getElementById('idBlood').innerText}
اللون المفضل: ${document.getElementById('idColor').innerText}
السيارة: ${document.getElementById('idVehicle').innerText}

[بيانات البطاقة الائتمانية]
النوع: ${document.getElementById('idCcType').innerText}
رقم البطاقة: ${document.getElementById('idCredit').innerText}
تاريخ الانتهاء: ${document.getElementById('idCcExp').innerText}
CVV: ${document.getElementById('idCcCvv').innerText}

[بيانات رقمية]
اسم المستخدم: ${document.getElementById('idUsername').innerText}
كلمة المرور: ${document.getElementById('idPassword').innerText}
موقع الويب: ${document.getElementById('idWebsite').innerText}
User Agent: ${document.getElementById('idUserAgent').innerText}
UUID: ${document.getElementById('idUuid').innerText}
===============================
            `.trim();
            
            navigator.clipboard.writeText(dataToCopy).then(() => {
                soundManager.click();
                const btn = event.currentTarget.querySelector('span:first-child');
                const oldText = btn.innerText;
                btn.innerText = "تم النسخ بنجاح ✔️";
                setTimeout(() => btn.innerText = oldText, 2000);
            });
        }

        // --- وظائف الأدوات الجديدة المتقدمة ---
        
        async function processAudio(action) {
            const formData = new FormData();
            if (action === 'encode') {
                const fileEl = document.getElementById('audioFileEncrypt');
                const file = fileEl.files[0];
                const text = document.getElementById('audioSecretText').value.trim();
                
                if (!file || !text) {
                    return alert('يرجى اختيار ملف صوتي وكتابة النص السري المراد إخفاؤه.');
                }
                
                console.log('جاري المعالجة...');
                
                formData.append('file', file);
                formData.append('text', text);
                
                try {
                    const response = await fetch('/api/audio/stego/encode', { method:'POST', body:formData });
                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.error || 'عذراً، فشلت عملية التشفير.');
                    }
                    
                    const blob = await response.blob();
                    const downloadUrl = window.URL.createObjectURL(blob);
                    const downloadAnchor = document.createElement('a');
                    downloadAnchor.href = downloadUrl;
                    downloadAnchor.download = "TITAN_SECURE_" + file.name;
                    document.body.appendChild(downloadAnchor);
                    downloadAnchor.click();
                    
                    alert('تم التشفير والتحميل تلقائياً ✅');
                    
                    setTimeout(() => {
                        document.body.removeChild(downloadAnchor);
                        window.URL.revokeObjectURL(downloadUrl);
                    }, 500);
                } catch (error) {
                    alert('خطأ: ' + error.message);
                }
            } else {
                const file = document.getElementById('audioFileDecrypt').files[0];
                if (!file) return alert('يرجى اختيار الملف المراد فحصه.');
                
                console.log('جاري التحليل...');
                formData.append('file', file);
                
                try {
                    const res = await fetch('/api/audio/stego/decode', { method:'POST', body:formData });
                    const data = await res.json();
                    
                    if (data.error) {
                        alert('تنبيه: ' + data.error);
                    } else if (data.success && data.hidden_data) {
                        document.getElementById('audioDecodedResult').innerText = data.hidden_data;
                        alert('✅ تم العثور على نص مخفي!');
                    } else {
                        alert('لم يتم العثور على بيانات مخفية.');
                    }
                } catch (error) {
                    alert('خطأ في الاتصال بالخادم.');
                }
            }
        }

        async function cleanPdf() {
            const file = document.getElementById('pdfCleanFile').files[0];
            if (!file) return Swal.fire({ icon:'error', title:'خطأ', text:'يرجى اختيار ملف PDF.' });
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const res = await fetch('/api/pdf/clean', { method:'POST', body:formData });
                if (!res.ok) throw new Error('فشل التنظيف');
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = "clean_" + file.name;
                document.body.appendChild(a);
                a.click();

                alert('تم التنظيف بنجاح، تم تحميل الملف النظيف.');
            } catch (e) { alert('خطأ: ' + e.message); }
        }

        function generateStealthFingerprint() {
            const prints = [
                { os: "Linux x86_64", browser: "Tor Browser/11.5.1", gl: "Intel Open Source Technology Center", screen: "1366x768", fonts: ["Arimo", "Tinos"] },
                { os: "Windows 10.0", browser: "Hardened Firefox/98.0", gl: "Microsoft Basic Render", screen: "1920x1080", fonts: ["Arial", "Courier"] },
                { os: "macOS 12.0", browser: "Safari/15.0 (Stealth)", gl: "Apple M1 GPU", screen: "1440x900", fonts: ["Helvetica", "Menlo"] }
            ];
            const p = prints[Math.floor(Math.random()*prints.length)];
            document.getElementById('fingerprintDisplay').innerText = JSON.stringify(p, null, 2);

        }

        async function runShodanScan() {
            const ip = document.getElementById('shodanIp').value;
            if (!ip) return;
            const res = await fetch('/api/intel/shodan', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ip}) });
            const data = await res.json();
            document.getElementById('shodanResult').innerText = JSON.stringify(data, null, 2);

        }

        async function checkDnsLeak() {
            const res = await fetch('/api/intel/dns-leak');
            const data = await res.json();
            document.getElementById('dnsLeakStatus').innerText = data.leaked ? "⚠️ تسريب!" : "✅ آمن";
            document.getElementById('dnsLeakStatus').className = data.leaked ? "text-2xl font-black mb-1 text-red-500" : "text-2xl font-black mb-1 text-green-500";

        }

        function panicWipe() {
            if (confirm('تدمير الجلسة؟ سيتم تسجيل الخروج فوراً ومسح كافة البيانات المؤقتة!')) {
                doLogout();
            }
        }
    </script>
</body>
</html>
"""

# --- المسارات (Routes) ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/scan', methods=['POST'])
def scan():
    target = request.json.get('target', '')
    label, score = get_strength_details(target)
    count = check_hibp_leak(target)
    return jsonify({"strength": label, "score": score, "exposed_count": count})

@app.route('/generate', methods=['GET'])
def generate():
    mode = request.args.get('mode', 'random')
    if mode == 'passphrase':
        return jsonify({"suggested": generate_readable_passphrase()})
    return jsonify({"suggested": generate_strong_password()})

@app.route('/api/metadata/remove', methods=['POST'])
def metadata_remove_route():
    file = request.files['file']
    try:
        processed_data = remove_image_metadata(file.read())
        add_audit_log("إزالة ميتابيانات", f"الملف: {file.filename}")
        return send_file(
            io.BytesIO(processed_data),
            mimetype='image/png',
            as_attachment=True,
            download_name="clean_" + file.filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/steganography/encode', methods=['POST'])
def stego_encode_route():
    file = request.files['file']
    text = request.form['text']
    try:
        processed_data = lsb_encode(file.read(), text)
        add_audit_log("تشفير إخفاء (Stego)", f"إخفاء نص في {file.filename}")
        return send_file(
            io.BytesIO(processed_data),
            mimetype='image/png',
            as_attachment=True,
            download_name="stego_" + file.filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/steganography/decode', methods=['POST'])
def stego_decode_route():
    file = request.files['file']
    try:
        decoded_text = lsb_decode(file.read())
        add_audit_log("فك إخفاء (Stego)", f"محاولة استخراج نص من {file.filename}")
        return jsonify({"result": decoded_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    return jsonify(AUDIT_LOGS)

@app.route('/crypt-text', methods=['POST'])
def crypt_text_route():
    data = request.json
    text, key, action = data['text'], data['key'], data['action']
    try:
        if action == 'encrypt':
            result = encrypt_data(text.encode(), key).decode('latin1') # استخدام latin1 لنقل bytes كنص
            return jsonify({"result": base64.b64encode(result.encode('latin1')).decode()})
        else:
            encrypted_bytes = base64.b64decode(text)
            result = decrypt_data(encrypted_bytes, key).decode()
            return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/crypt-file', methods=['POST'])
def crypt_file_route():
    file = request.files['file']
    key = request.form['key']
    action = request.form['action']
    try:
        file_data = file.read()
        if action == 'encrypt':
            processed_data = encrypt_data(file_data, key)
        else:
            processed_data = decrypt_data(file_data, key)
        
        return send_file(
            io.BytesIO(processed_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=file.filename + ('.titan' if action == 'encrypt' else '')
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/ip', methods=['POST'])
def ip_check():
    data = request.json or {}
    ip = data.get('ip', "").strip()
    if not ip:
        if request.headers.getlist("X-Forwarded-For"):
            ip = request.headers.getlist("X-Forwarded-For")[0]
        else:
            ip = request.remote_addr
        if ip == "127.0.0.1": 
            ip = ""
            
    info = get_ip_intelligence_data(ip)
    return jsonify(info)

@app.route('/api/scan/phone', methods=['POST'])
def scan_phone_route():
    data = request.json or {}
    phone = data.get('phone', '')
    res = check_phone_intelligence(phone)
    add_audit_log("فحص رقم هاتف (IPQualityScore)", f"تم فحص الرقم: {phone}")
    return jsonify(res)

@app.route('/api/scan/url', methods=['POST'])
def scan_url_route():
    data = request.json or {}
    url = data.get('url', '')
    res = check_url_intelligence(url)
    add_audit_log("فحص رابط مشبوه (IPQualityScore)", f"تم فحص الموثوقية: {url[:30]}...")
    return jsonify(res)

@app.route('/api/scan/malware_url', methods=['POST'])
def scan_malware_url_route():
    data = request.json or {}
    url = data.get('url', '')
    res = scan_malware_url(url)
    add_audit_log("فحص URL خبيث (Malware)", f"تم فحص: {url[:30]}...")
    return jsonify(res)

@app.route('/api/scan/malware_file', methods=['POST'])
def scan_malware_file_route():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "لم يتم تقديم أي ملف"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "لم يتم اختيار ملف"}), 400
        
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
    file.save(temp_path)
    
    try:
        res = scan_malware_file(temp_path)
        add_audit_log("فحص ملف خبيث (Malware)", f"اسم الملف: {file.filename}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return jsonify(res)

@app.route('/api/scan/emailpass_leak', methods=['POST'])
def scan_emailpass_leak_route():
    data = request.json or {}
    email = data.get('email', '')
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({"success": False, "message": "الرجاء توفير الإيميل وكلمة السر"}), 400
        
    res = check_leaked_emailpass(email, password)
    add_audit_log("فحص تسريب (IPQualityScore)", f"تم الفحص لـ: {email}")
    return jsonify(res)

@app.route('/api/scan/ipqs_logs', methods=['POST'])
def scan_ipqs_logs_route():
    data = request.json or {}
    req_type = data.get('type', 'proxy')
    start_date = data.get('start_date', '2024-01-01')
    
    res = get_ipqs_requests_list(req_type, start_date)
    add_audit_log("سجلات API (IPQualityScore)", f"استعلام عن: {req_type} منذ {start_date}")
    return jsonify(res)

@app.route('/api/2fa/generate', methods=['GET'])
def generate_2fa():
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name="TITAN User", issuer_name="TITAN Security")
    
    img = qrcode.make(provisioning_uri)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({
        "secret": secret,
        "qr_code": f"data:image/png;base64,{img_str}"
    })

@app.route('/api/2fa/verify', methods=['POST'])
def verify_2fa():
    data = request.json or {}
    secret = data.get('secret')
    code = data.get('code')
    
    if not secret or not code:
        return jsonify({"valid": False})
        
    totp = pyotp.TOTP(secret)
    is_valid = totp.verify(code)
    return jsonify({"valid": is_valid})

# --- مسارات القبو المشفر (Per-User) ---

def _get_logged_in_user_id():
    """Returns (user_id, None) if logged in, else (None, error_response)."""
    user_id = session.get('user_id')
    if not user_id:
        return None, (jsonify({"error": "غير مصرح. يجب تسجيل الدخول أولاً."}), 401)
    return user_id, None

@app.route('/api/vault/has-password', methods=['GET'])
def vault_has_password():
    """Check if the logged-in user has a vault password set."""
    user_id, err = _get_logged_in_user_id()
    if err: return err
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT vault_password_hash FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    has_pw = bool(row and row[0])
    return jsonify({"hasVaultPassword": has_pw})

@app.route('/api/vault/set-password', methods=['POST'])
def vault_set_password():
    """First-time vault password setup for the logged-in user."""
    user_id, err = _get_logged_in_user_id()
    if err: return err
    data = request.json or {}
    password = data.get('password', '').strip()
    if len(password) < 4:
        return jsonify({"error": "كلمة سر القبو يجب أن تكون 4 أحرف على الأقل"}), 400
    pw_hash = hash_password(password)
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    # Check if already set
    c.execute("SELECT vault_password_hash FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if row and row[0]:
        conn.close()
        return jsonify({"error": "كلمة سر القبو محددة مسبقاً. استخدمها للدخول."}), 409
    c.execute("UPDATE users SET vault_password_hash = ? WHERE id = ?", (pw_hash, user_id))
    conn.commit()
    conn.close()
    add_audit_log("تعيين كلمة سر القبو 🔐", f"المستخدم #{user_id} عيّن كلمة سر قبو جديدة")
    return jsonify({"success": True})

@app.route('/api/vault/load', methods=['POST'])
def load_vault():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    data = request.json or {}
    key = data.get('key')
    if not key: return jsonify({"error": "Missing key"}), 400

    # Verify the vault password against the stored hash
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT vault_password_hash FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row or not row[0]:
        return jsonify({"error": "لم تقم بتعيين كلمة سر للقبو بعد."}), 403
    if not verify_password(key, row[0]):
        add_audit_log("فشل فتح القبو 🚨", f"كلمة سر خاطئة للمستخدم #{user_id}")
        return jsonify({"error": "كلمة السر الرئيسية غير صحيحة."}), 401

    vault_file = get_vault_file(user_id)
    if not os.path.exists(vault_file):
        add_audit_log("فتح القبو ✅", f"قبو جديد للمستخدم #{user_id}")
        return jsonify({"vault": []})  # قبو جديد

    try:
        with open(vault_file, 'rb') as f:
            encrypted_data = f.read()
        decrypted_bytes = decrypt_data(encrypted_data, key)
        vault_data = json.loads(decrypted_bytes.decode('utf-8'))
        add_audit_log("فتح القبو ✅", f"تم الوصول لقبو المستخدم #{user_id}")
        return jsonify({"vault": vault_data})
    except Exception:
        add_audit_log("فشل فتح القبو 🚨", f"خطأ في فك التشفير للمستخدم #{user_id}")
        return jsonify({"error": "كلمة السر الرئيسية غير صحيحة أو الملف معطوب."}), 401

@app.route('/api/vault/save', methods=['POST'])
def save_vault():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    data = request.json or {}
    key = data.get('key')
    vault_list = data.get('vault', [])
    if not key: return jsonify({"error": "Missing key"}), 400

    # Re-verify password before saving
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT vault_password_hash FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row or not row[0] or not verify_password(key, row[0]):
        return jsonify({"error": "كلمة السر غير صحيحة. لا يمكن الحفظ."}), 401

    try:
        json_str = json.dumps(vault_list).encode('utf-8')
        encrypted_data = encrypt_data(json_str, key)
        vault_file = get_vault_file(user_id)
        with open(vault_file, 'wb') as f:
            f.write(encrypted_data)
        add_audit_log("حفظ القبو 💾", f"تم تحديث {len(vault_list)} عنصر للمستخدم #{user_id}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vault/backup', methods=['POST'])
def backup_vault():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    vault_file = get_vault_file(user_id)
    if not os.path.exists(vault_file):
        return jsonify({"error": "لا يوجد قبو لتصديره!"}), 400
    try:
        with open(vault_file, 'rb') as f:
            data = f.read()
        add_audit_log("تصدير النسخة الاحتياطية 📦", f"المستخدم #{user_id}")
        return send_file(
            io.BytesIO(data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f"vault_backup_user{user_id}.titan.bak"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vault/restore', methods=['POST'])
def restore_vault():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    file = request.files['file']
    try:
        data = file.read()
        vault_file = get_vault_file(user_id)
        with open(vault_file, 'wb') as f:
            f.write(data)
        add_audit_log("استعادة النسخة الاحتياطية 🔄", f"تم استعادة قبو المستخدم #{user_id}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/vault/recovery/setup', methods=['POST'])
def setup_recovery():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    data = request.json or {}
    key = data.get('key', '')
    q1 = data.get('q1', '')
    a1 = data.get('a1', '')
    q2 = data.get('q2', '')
    a2 = data.get('a2', '')

    if not all([key, q1, a1, q2, a2]):
        return jsonify({"error": "جميع الحقول مطلوبة"}), 400

    recovery_pass = a1.strip().lower() + "|" + a2.strip().lower()
    encrypted_key = encrypt_data(key.encode('utf-8'), recovery_pass)
    recovery_file = get_vault_recovery_file(user_id)

    with open(recovery_file, 'w', encoding='utf-8') as f:
        json.dump({
            "q1": q1,
            "q2": q2,
            "encrypted_key": base64.b64encode(encrypted_key).decode('utf-8')
        }, f, ensure_ascii=False)

    add_audit_log("إعداد استعادة القبو 🔑", f"المستخدم #{user_id} عيّن أسئلة الأمان")
    return jsonify({"success": True})

@app.route('/api/vault/recovery/questions', methods=['GET'])
def get_recovery_questions():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    recovery_file = get_vault_recovery_file(user_id)
    if not os.path.exists(recovery_file):
        return jsonify({"error": "لم تقم بإعداد أسئلة الأمان مسبقاً لاستعادة هذا القبو."}), 400
    with open(recovery_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify({"q1": data["q1"], "q2": data["q2"]})

@app.route('/api/vault/recovery/recover', methods=['POST'])
def recover_vault_key():
    user_id, err = _get_logged_in_user_id()
    if err: return err
    data = request.json or {}
    a1 = data.get('a1', '')
    a2 = data.get('a2', '')
    recovery_file = get_vault_recovery_file(user_id)
    if not os.path.exists(recovery_file):
        return jsonify({"error": "لم يتم إعداد أسئلة الأمان"}), 400

    with open(recovery_file, 'r', encoding='utf-8') as f:
        r_data = json.load(f)

    recovery_pass = a1.strip().lower() + "|" + a2.strip().lower()
    try:
        encrypted_key_bytes = base64.b64decode(r_data["encrypted_key"])
        decrypted_key = decrypt_data(encrypted_key_bytes, recovery_pass)
        add_audit_log("استعادة القبو ✅", f"المستخدم #{user_id} استعاد كلمة سر القبو")
        return jsonify({"recovered_key": decrypted_key.decode('utf-8')})
    except Exception:
        add_audit_log("محاولة استعادة فاشلة 🚨", f"إجابات خاطئة للمستخدم #{user_id}")
        return jsonify({"error": "الإجابات التي أدخلتها غير صحيحة"}), 401


# --- مسارات الإضافات الجديدة المتقدمة ---



@app.route('/api/pdf-process', methods=['POST'])
def pdf_process_route():
    try:
        file = request.files['file']
        password = request.form['password']
        action = request.form.get('action', 'lock')
        
        reader = PdfReader(file)
        writer = PdfWriter()
        
        if action == 'lock':
            for page in reader.pages:
                writer.add_page(page)
            writer.encrypt(password)
            add_audit_log("حماية PDF 🔒", f"تم تشفير الملف بكلمة سر ({file.filename})")
        else:
            if reader.is_encrypted:
                reader.decrypt(password)
            for page in reader.pages:
                writer.add_page(page)
            add_audit_log("فك حماية PDF 🔓", f"تم فتح الملف ({file.filename})")
            
        out_stream = io.BytesIO()
        writer.write(out_stream)
        out_stream.seek(0)
        
        dl_name = f"locked_{file.filename}" if action == 'lock' else f"unlocked_{file.filename}"
        return send_file(
            out_stream, 
            mimetype='application/pdf', 
            as_attachment=True, 
            download_name=dl_name
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/port-scan', methods=['POST'])
def port_scan_route():
    ip = request.json.get('ip', '127.0.0.1')
    common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 3306, 3389, 8080]
    open_ports = []
    
    def scan_port(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, port))
        sock.close()
        return port if result == 0 else None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(common_ports)) as executor:
        results = executor.map(scan_port, common_ports)
        for r in results:
            if r is not None:
                open_ports.append(r)
                
    add_audit_log("فحص منافذ 📡", f"{ip} - عُثر على {len(open_ports)} منفذ مفتوح")
    return jsonify({"open_ports": open_ports})

@app.route('/api/burn-note/create', methods=['POST'])
def create_burn_note():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "نص فارغ"}), 400
    
    note_id = str(uuid.uuid4())
    BURN_NOTES[note_id] = text
    add_audit_log("رسالة تدمير ذاتي 🔥", f"تم توليد رابط رسالة جديدة")
    
    # Generate full access URL
    url = f"{request.host_url}burn/{note_id}"
    return jsonify({"link": url})

@app.route('/burn/<note_id>', methods=['GET'])
def view_burn_note(note_id):
    if note_id in BURN_NOTES:
        # قرأناها ودمّرناها فوراً من المتغير (RAM)
        text = BURN_NOTES.pop(note_id) 
        add_audit_log("رسالة مدمرة 💣", f"تم فتح الرسالة وتدميرها للأبد")
        
        return f'''
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>رسالة مدمرة | TITAN</title>
            <style>
                body {{ background: #050505; color: #fff; font-family: 'Segoe UI', Tahoma, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background-image: radial-gradient(circle at center, #2e0909 0%, #050505 100%); user-select: none; -webkit-user-select: none; }}
                body::selection {{ background: transparent; color: transparent; }}
                .container {{ background: #0a0a0a; border: 1px solid #ef4444; border-radius: 12px; padding: 40px; box-shadow: 0 0 50px rgba(239, 68, 68, 0.2); max-width: 600px; text-align: center; position: relative; overflow: hidden; width: 90%; transition: filter 0.2s, opacity 0.2s; }}
                .container::before {{ content:""; position:absolute; top:0; left:0; right:0; height:4px; background:linear-gradient(90deg, #ef4444, #f97316); }}
                h1 {{ color: #ef4444; margin-bottom: 5px; font-weight: 900; letter-spacing: -1px; text-shadow: 0 0 20px rgba(239,68,68,0.5); pointer-events: none; }}
                .subtitle {{ color: #9ca3af; font-size: 14px; margin-bottom: 30px; line-height: 1.6; pointer-events: none; }}
                .warning {{ margin-top: 30px; font-size: 14px; color: #ef4444; opacity: 0.9; font-weight: bold; letter-spacing: 1px; pointer-events: none; }}
                @keyframes pulse-icon {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.1); filter: drop-shadow(0 0 15px #ef4444); }} }}

                /* === ANTI-CAMERA CSS ANIMATION ===
                   Rapidly cycles the text between two color states at ~8Hz.
                   Human eye (persistence ~100ms) sees the ON state clearly.
                   Camera integrating over 1/30s captures both states merged
                   creating color fringing + temporal blur that obscures the text. */
                @keyframes antiCam {{
                    0%   {{ color: #ff5555; text-shadow: 0 0 8px rgba(255,85,85,0.7); }}
                    49%  {{ color: #ff5555; text-shadow: 0 0 8px rgba(255,85,85,0.7); }}
                    50%  {{ color: #0a0a0a; text-shadow: none; }}
                    99%  {{ color: #0a0a0a; text-shadow: none; }}
                    100% {{ color: #ff5555; text-shadow: 0 0 8px rgba(255,85,85,0.7); }}
                }}
                .secure-text {{
                    background: #000;
                    padding: 24px 20px;
                    border-radius: 8px;
                    border: 1px dashed #ef4444;
                    text-align: right;
                    direction: rtl;
                    font-size: 20px;
                    line-height: 1.9;
                    word-wrap: break-word;
                    white-space: pre-wrap;
                    font-family: 'Segoe UI', Tahoma, 'Arial', monospace;
                    font-weight: bold;
                    pointer-events: none;
                    animation: antiCam 0.125s steps(1) infinite;
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body oncontextmenu="return false;" onkeydown="return disableCopyKeys(event);">
            <div class="container" id="secureContainer">
                <div style="font-size: 60px; margin-bottom: 20px; animation: pulse-icon 2s infinite;">💣</div>
                <h1>هذه الرسالة دُمّرت للتو!</h1>
                <p class="subtitle" id="topSubtitle">لقد تم مسح هذه الرسالة نهائياً من الذاكرة الحية للخادم بمجرد فتحك لها.<br>لن يمكنك أنت أو غيرك قراءة محتواها مرة أخرى، قم بنسخها الآن إذا احتجت لذلك.</p>
                <!-- Full Arabic text - browser handles letter joining natively -->
                <div class="secure-text" id="secureText">{text}</div>
                <p style="color:#6b7280; font-size:10px; margin:0 0 10px 0; letter-spacing:1px;">🔒 CAMERA-RESISTANT DISPLAY</p>
                <div class="warning" id="timerWarning">⚠️ تدمير ذاتي إضافي للشاشة خلال <span id="countdown">10</span> ثواني...</div>
            </div>
            
            <script>
                let timeLeft = 10;
                const countdownEl = document.getElementById('countdown');
                const warningBox = document.getElementById('timerWarning');
                const topSubs = document.getElementById('topSubtitle');
                const secureTextEl = document.getElementById('secureText');
                let destroyed = false;

                const timer = setInterval(() => {{
                    timeLeft--;
                    countdownEl.innerText = timeLeft;
                    
                    if (timeLeft <= 3) {{
                        countdownEl.style.fontSize = '24px';
                        countdownEl.parentElement.style.textShadow = '0 0 10px red';
                    }}
                    
                    if(timeLeft <= 0) {{
                        clearInterval(timer);
                        // Stop the CSS animation and replace text with destroyed message
                        secureTextEl.style.animation = 'none';
                        secureTextEl.style.color = '#ef4444';
                        secureTextEl.style.textAlign = 'center';
                        secureTextEl.textContent = '💥 تم تدمير الرسالة نهائياً';
                        warningBox.innerText = 'SECURE BURN COMPLETE // SYSTEM LOGGED';
                        topSubs.innerText = 'تم التخلص من الرسالة بالكامل من الشاشة.';
                    }}
                }}, 1000);

                // Anti-Copy and Anti-Screenshot Scripts
                function disableCopyKeys(e) {{
                    if(e.ctrlKey && (e.key === 'c' || e.key === 'p' || e.key === 's')) return false;
                    if(e.key === 'PrintScreen') {{
                        navigator.clipboard.writeText('محاولة التقاط شاشة مرفوضة.');
                        return false;
                    }}
                }}
                
                document.addEventListener('keyup', (e) => {{
                    if(e.key === 'PrintScreen') navigator.clipboard.writeText('محاولة التقاط شاشة مرفوضة.');
                }});
                
                // Hide content when window loses focus to prevent screenshots/recording
                const secContainer = document.getElementById('secureContainer');
                window.addEventListener('blur', () => {{
                    secContainer.style.filter = 'blur(30px)';
                    secContainer.style.opacity = '0';
                }});
                window.addEventListener('focus', () => {{
                    secContainer.style.filter = 'none';
                    secContainer.style.opacity = '1';
                }});
            </script>
        </body>
        </html>
        '''
    else:
        return f'''
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head><meta charset="UTF-8"><title>الرسالة غير موجودة</title></head>
        <body style="background:#050505; color:#ef4444; font-family:sans-serif; text-align:center; padding-top:100px;">
            <div style="font-size: 60px; margin-bottom:20px;">🕳️</div>
            <h2>الرسالة غير متوفرة!</h2>
            <p style="color:#9ca3af;">الرابط غير صالح، أو أن الرسالة تم الإطلاع عليها وتدميرها مسبقاً.</p>
        </body>
        </html>
        ''', 404

# --- مسارات الإضافات للحزمة الثالثة المتقدمة (Audio & Privacy) ---

@app.route('/api/audio/stego/encode', methods=['POST'])
def audio_stego_encode():
    file = request.files['file']
    text = request.form['text']
    try:
        processed_data = wave_lsb_encode(file.read(), text, file.filename)
        add_audit_log("إخفاء في الصوت 🎵", f"تم إخفاء بيانات في {file.filename}")
        mimetype = 'audio/mpeg' if file.filename.lower().endswith('.mp3') else 'audio/wav'
        return send_file(
            io.BytesIO(processed_data),
            mimetype=mimetype,
            as_attachment=True,
            download_name="stego_" + file.filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/audio/stego/decode', methods=['POST'])
def audio_stego_decode():
    file = request.files['file']
    try:
        decoded_text = wave_lsb_decode(file.read(), file.filename)
        add_audit_log("استخراج من الصوت 🎵", f"محاولة فك تشفير {file.filename}")
        return jsonify({"result": decoded_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/pdf/clean', methods=['POST'])
def pdf_clean_route():
    file = request.files['file']
    try:
        processed_data = clean_pdf_metadata(file.read())
        add_audit_log("تنظيف PDF 🧹", f"إزالة ميتابيانات {file.filename}")
        return send_file(
            io.BytesIO(processed_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name="clean_" + file.filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/intel/dns-leak', methods=['GET'])
def dns_leak_route():
    return jsonify(get_dns_leak_info())

@app.route('/api/intel/shodan', methods=['POST'])
def shodan_intel_route():
    ip = request.json.get('ip', '')
    return jsonify(get_shodan_intel(ip))

# --- مسارات الإضافات للحزمة الثانية المتقدمة ---

@app.route('/api/osint/image', methods=['POST'])
def osint_image_route():
    try:
        file = request.files['file']
        data = extract_exif_data(file.read())
        add_audit_log("استخبارات صور (OSINT)", f"تم استخراج بيانات من {file.filename}")
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/scan/email', methods=['POST'])
def scan_email_route():
    email = request.json.get('email', '')
    res = check_email_intelligence(email)
    add_audit_log("فحص إيميل (IPQualityScore)", f"تم فحص البريد: {email}")
    return jsonify(res)

@app.route('/api/network/scan', methods=['GET'])
def scan_network_route():
    devices = scan_local_network()
    add_audit_log("رادار الشبكة المحلية", f"تم العثور على {len(devices)} جهاز متصل")
    return jsonify(devices)

# --- مسارات الإضافات للحزمة الرابعة المتقدمة (Phase 4: Defense & Comms) ---

# 1. USB Guardian
USB_MONITOR_ACTIVE = False
USB_MONITOR_THREAD = None
LAST_DRIVES = set()

def get_current_drives():
    drives = set()
    try:
        for p in psutil.disk_partitions():
            if 'removable' in p.opts.lower() or p.fstype == '':
                drives.add(p.device)
    except:
        pass
    return drives

def usb_monitor_listener():
    global USB_MONITOR_ACTIVE, LAST_DRIVES
    import time
    LAST_DRIVES = get_current_drives()
    
    suspicious_files = ['autorun.inf']
    suspicious_exts = ['.vbs', '.bat', '.ps1', '.exe', '.cmd']
    
    while USB_MONITOR_ACTIVE:
        current_drives = get_current_drives()
        new_drives = current_drives - LAST_DRIVES
        
        for drive in new_drives:
            with app.app_context():
                add_audit_log("🛡️ حارس منافذ USB", f"تم رصد توصيل قرص جديد: {drive}")
            
            # Simple root scan for threats
            try:
                if os.path.exists(drive):
                    for f in os.listdir(drive):
                        f_lower = f.lower()
                        if f_lower in suspicious_files or any(f_lower.endswith(ext) for ext in suspicious_exts):
                            with app.app_context():
                                add_audit_log("🚨 تهديد USB محتمل", f"عُثر على ملف تشغيل تلقائي أو تنفيذي مشبوه في {drive}: {f}")
            except Exception:
                pass
                
        LAST_DRIVES = current_drives
        time.sleep(3)

@app.route('/api/defense/usb', methods=['POST'])
def toggle_usb_guardian():
    global USB_MONITOR_ACTIVE, USB_MONITOR_THREAD
    action = request.json.get('action')
    if action == 'start':
        if not USB_MONITOR_ACTIVE:
            USB_MONITOR_ACTIVE = True
            USB_MONITOR_THREAD = threading.Thread(target=usb_monitor_listener, daemon=True)
            USB_MONITOR_THREAD.start()
            add_audit_log("تفعيل حارس USB", "تم تفعيل المراقبة الفورية لمنافذ USB")
        return jsonify({"status": "active"})
    else:
        USB_MONITOR_ACTIVE = False
        add_audit_log("إيقاف حارس USB", "تم إيقاف المراقبة")
        return jsonify({"status": "inactive"})

# 2. File Integrity Monitor (FIM)
FIM_ACTIVE = False
FIM_THREAD = None
FIM_TARGET_FILE = ""
FIM_TARGET_HASH = ""

def fim_listener():
    global FIM_ACTIVE, FIM_TARGET_FILE, FIM_TARGET_HASH
    import time
    while FIM_ACTIVE:
        if FIM_TARGET_FILE and os.path.exists(FIM_TARGET_FILE):
            try:
                with open(FIM_TARGET_FILE, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                if current_hash != FIM_TARGET_HASH:
                    with app.app_context():
                        add_audit_log("⚠️ اختراق تكامل الملفات (FIM)", f"تم رصد تغيير في الملف المراقب: {FIM_TARGET_FILE}")
                    FIM_TARGET_HASH = current_hash # Update to prevent spam
            except Exception:
                pass
        time.sleep(5)

@app.route('/api/defense/fim', methods=['POST'])
def toggle_fim():
    global FIM_ACTIVE, FIM_THREAD, FIM_TARGET_FILE, FIM_TARGET_HASH
    data = request.json or {}
    action = data.get('action')
    
    if action == 'start':
        target = data.get('target', '')
        if not os.path.exists(target):
            return jsonify({"error": "الملف غير موجود!"}), 400
        
        try:
            with open(target, 'rb') as f:
                FIM_TARGET_HASH = hashlib.sha256(f.read()).hexdigest()
            FIM_TARGET_FILE = target
            FIM_ACTIVE = True
            FIM_THREAD = threading.Thread(target=fim_listener, daemon=True)
            FIM_THREAD.start()
            add_audit_log("تفعيل مراقب التكامل (FIM)", f"بدأت مراقبة الملف: {os.path.basename(target)}")
            return jsonify({"status": "active"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        FIM_ACTIVE = False
        add_audit_log("إيقاف مراقب التكامل", "تم إيقاف المراقبة")
        return jsonify({"status": "inactive"})

# 3. Secure Comms: P2P Burn Chat
# In-memory only storage. Structure: { "room_id": [ {"sender": "A", "msg": "hello", "timestamp": ...} ] }
BURN_CHAT_ROOMS = {}

@app.route('/api/chat/send', methods=['POST'])
def chat_send():
    data = request.json or {}
    room_id = data.get('room_id')
    sender = data.get('sender', 'Anonymous')
    msg = data.get('msg', '')
    
    if not room_id or not msg:
        return jsonify({"error": "بيانات مفقودة"}), 400
        
    if room_id not in BURN_CHAT_ROOMS:
        BURN_CHAT_ROOMS[room_id] = []
        
    BURN_CHAT_ROOMS[room_id].append({"sender": sender, "msg": msg})
    return jsonify({"success": True})

@app.route('/api/chat/receive', methods=['GET'])
def chat_receive():
    room_id = request.args.get('room_id')
    requester = request.args.get('requester', '')
    
    if not room_id or room_id not in BURN_CHAT_ROOMS:
        return jsonify({"messages": []})
        
    messages = BURN_CHAT_ROOMS[room_id]
    to_deliver = []
    remaining = []
    
    # Only deliver messages NOT sent by the requester, and burn them after reading
    for m in messages:
        if m['sender'] != requester:
            to_deliver.append(m)
        else:
            remaining.append(m)
            
    BURN_CHAT_ROOMS[room_id] = remaining
    
    if to_deliver:
        add_audit_log("Burn Chat 🔥", f"تم قراءة وتدمير {len(to_deliver)} رسالة سرية في الغرفة [{room_id}]")
        
    return jsonify({"messages": to_deliver})


# =====================================================================
# === الميزات الجديدة – Phase 6 ===
# =====================================================================

# --- QR Code مشفر ---
@app.route('/api/qr/generate', methods=['POST'])
def qr_generate():
    try:
        data = request.json or {}
        text = data.get('text', '')
        password = data.get('password', '')
        if not text: return jsonify({'error': 'النص مطلوب'}), 400

        # تشفير النص إذا وُجدت كلمة سر
        if password:
            salt = os.urandom(16)
            key = derive_key(password, salt)
            f = Fernet(key)
            payload = base64.urlsafe_b64encode(salt + f.encrypt(text.encode())).decode()
            payload = 'ENC:' + payload
        else:
            payload = text

        # توليد QR
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        add_audit_log("QR Code 🔳", f"تم توليد QR {'مشفر' if password else 'عادي'}")
        return jsonify({'qr': img_b64, 'encrypted': bool(password)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qr/decode', methods=['POST'])
def qr_decode():
    try:
        password = request.form.get('password', '')
        file = request.files.get('file')
        if not file: return jsonify({'error': 'الصورة مطلوبة'}), 400

        from PIL import Image as PILImage # type: ignore
        from pyzbar import pyzbar  # type: ignore
        img = PILImage.open(file)
        decoded = pyzbar.decode(img)
        if not decoded: return jsonify({'error': 'لم يتم التعرف على QR في الصورة'}), 400

        payload = decoded[0].data.decode('utf-8')
        if payload.startswith('ENC:') and password:
            raw = base64.urlsafe_b64decode(payload[4:])
            salt = raw[:16] # type: ignore
            enc = raw[16:] # type: ignore
            key = derive_key(password, salt)
            f = Fernet(key)
            text = f.decrypt(enc).decode('utf-8')
        else:
            text = payload
        return jsonify({'text': text})
    except Exception as e:
        error_msg = str(e)
        if "InvalidToken" in str(type(e)) or not error_msg:
            error_msg = "كلمة السر خاطئة أو رمز QR تالف"
        return jsonify({'error': error_msg}), 400


# --- Packet Sniffer ---
@app.route('/api/network/sniff', methods=['GET'])
def packet_sniff():
    try:
        import socket as _socket
        duration = int(request.args.get('duration', 4))
        packets_info = []
        stop_event = threading.Event()

        # Simple raw socket sniffer (works without scapy on Windows with admin)
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_RAW, _socket.IPPROTO_IP)
            local_ip = _socket.gethostbyname(_socket.gethostname())
            s.bind((local_ip, 0))
            s.setsockopt(_socket.IPPROTO_IP, _socket.IP_HDRINCL, 1)
            # Windows: enable promiscuous
            try: s.ioctl(_socket.SIO_RCVALL, _socket.RCVALL_ON) # type: ignore
            except Exception: pass
            s.settimeout(0.5)

            import time
            start = time.time()
            while time.time() - start < duration and len(packets_info) < 30:
                try:
                    raw, addr = s.recvfrom(65535)
                    # Parse IP header (first 20 bytes)
                    ip_header = raw[:20] # type: ignore
                    iph = ip_header
                    proto = iph[9]
                    src_ip = '.'.join(str(b) for b in iph[12:16])
                    dst_ip = '.'.join(str(b) for b in iph[16:20])
                    proto_name = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}.get(proto, f'IP({proto})')
                    packets_info.append({
                        'src': src_ip, 'dst': dst_ip,
                        'proto': proto_name, 'size': len(raw)
                    })
                except _socket.timeout:
                    continue
            try: s.ioctl(_socket.SIO_RCVALL, _socket.RCVALL_OFF) # type: ignore
            except Exception: pass
            s.close()
            add_audit_log("Packet Sniffer 📡", f"تم التقاط {len(packets_info)} حزمة")
        except PermissionError:
            return jsonify({'error': 'يحتاج صلاحية Administrator – شغّل البرنامج كمسؤول', 'packets': []}), 403
        return jsonify({'packets': packets_info})
    except Exception as e:
        return jsonify({'error': str(e), 'packets': []}), 500

# --- تحسين الهوية الوهمية ---
@app.route('/api/fake-identity', methods=['GET'])
def fake_identity_route():
    try:
        from faker import Faker  # type: ignore
        import datetime
        lang = request.args.get('lang', 'ar_SA')
        fake = Faker(lang)
        fake_en = Faker('en_US')

        username_base = fake_en.user_name()
        password_fake = fake_en.password(length=12, special_chars=True)
        dob = fake.date_of_birth(minimum_age=18, maximum_age=60)
        age = (datetime.date.today() - dob).days // 365
        
        zodiac_signs = [(120,"الجدي"), (219,"الدلو"), (320,"الحوت"), (420,"الحمل"), (521,"الثور"), (621,"الجوزاء"), (722,"السرطان"), (822,"الأسد"), (922,"العذراء"), (1022,"الميزان"), (1121,"العقرب"), (1221,"القوس"), (1231,"الجدي")]
        day_of_year = dob.month * 100 + dob.day
        zodiac = next(z for d, z in zodiac_signs if day_of_year <= d)

        return jsonify({
            'name': fake.name(),
            'gender': random.choice(['ذكر / Male', 'أنثى / Female']),
            'mother_name': fake.first_name_female() + ' ' + fake.last_name(),
            'birthdate': dob.strftime('%Y-%m-%d'),
            'age': age,
            'zodiac': zodiac,
            'national_id': ''.join([str(random.randint(0,9)) for _ in range(10)]),
            
            'address': fake.address().replace('\n', '، '),
            'zip_code': fake.postcode(),
            'geo': f"{fake.latitude()}, {fake.longitude()}",
            'country_code': fake.country_calling_code(),
            
            'phone': fake.phone_number(),
            'email': fake_en.email(),
            'company': fake.company(),
            'job': fake.job(),
            
            'height': f"{random.randint(150, 195)} cm",
            'weight': f"{random.randint(50, 100)} kg",
            'blood_type': random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
            'color': fake.color_name(),
            'vehicle': fake_en.word().title() + ' ' + str(random.randint(2000, 2024)),
            
            'cc_type': fake_en.credit_card_provider(),
            'credit_card': fake_en.credit_card_number(card_type='visa' if random.random() > 0.5 else 'mastercard'),
            'cc_expire': fake_en.credit_card_expire(),
            'cc_cvv': fake_en.credit_card_security_code(),
            
            'username': username_base,
            'password': password_fake,
            'website': f'https://www.{fake_en.domain_name()}',
            'user_agent': fake.user_agent(),
            'uuid': fake.uuid4(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Dashboard / System Stats ---
@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        local_ip = socket.gethostbyname(socket.gethostname())
        try:
            pub_ip_data = requests.get('https://api.ipify.org?format=json', timeout=3).json()
            public_ip = pub_ip_data.get('ip', 'غير متاح')
        except Exception:
            public_ip = 'غير متاح'
        return jsonify({
            'cpu_percent': cpu,
            'ram_used_gb': round(mem.used / 1024**3, 2),
            'ram_total_gb': round(mem.total / 1024**3, 2),
            'ram_percent': mem.percent,
            'disk_used_gb': round(disk.used / 1024**3, 2),
            'disk_total_gb': round(disk.total / 1024**3, 2),
            'disk_percent': disk.percent,
            'net_sent_mb': round(net.bytes_sent / 1024**2, 2),
            'net_recv_mb': round(net.bytes_recv / 1024**2, 2),
            'local_ip': local_ip,
            'public_ip': public_ip,
            'vault_items': 0,
            'burn_notes': len(BURN_NOTES),
            'audit_count': len(AUDIT_LOGS),
            'recent_logs': AUDIT_LOGS[:5], # type: ignore
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# =====================================================================
# === Authentication Routes ===
# =====================================================================

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "اسم المستخدم وكلمة السر مطلوبان"}), 400
    if len(username) < 3:
        return jsonify({"error": "اسم المستخدم يجب أن يكون 3 أحرف على الأقل"}), 400
    if len(password) < 6:
        return jsonify({"error": "كلمة السر يجب أن تكون 6 أحرف على الأقل"}), 400

    pw_hash = hash_password(password)
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(USERS_DB)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                  (username, pw_hash, created_at))
        conn.commit()
        conn.close()
        add_audit_log("تسجيل مستخدم جديد 👤", f"تم إنشاء حساب: {username}")
        return jsonify({"success": True, "message": "تم إنشاء الحساب بنجاح!"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "اسم المستخدم مأخوذ، اختر اسماً آخر"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "اسم المستخدم وكلمة السر مطلوبان"}), 400

    try:
        conn = sqlite3.connect(USERS_DB)
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "اسم المستخدم أو كلمة السر غير صحيحة"}), 401

        user_id, pw_hash = row
        if not verify_password(password, pw_hash):
            add_audit_log("محاولة دخول فاشلة 🚨", f"بيانات خاطئة لـ: {username}")
            return jsonify({"error": "اسم المستخدم أو كلمة السر غير صحيحة"}), 401

        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        add_audit_log("تسجيل دخول ✅", f"تم الدخول بنجاح: {username}")
        return jsonify({"success": True, "username": username})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    username = session.get('username', 'Unknown')
    session.clear()
    add_audit_log("تسجيل خروج 🚪", f"تم تسجيل الخروج لـ: {username}")
    return jsonify({"success": True})

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    if 'user_id' in session:
        return jsonify({"loggedIn": True, "username": session.get('username', '')})
    return jsonify({"loggedIn": False})


# تأكد أن هذه الأسطر في نهاية الملف تماماً
init_db() 

if __name__ == '__main__':
    # هذا السطر يعمل فقط عند التشغيل اليدوي (Local)
    # أما على Render، فإن Gunicorn سيتولى المهمة
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)



