"""Network utilities for checking ports and connections."""

import socket

def is_port_available(host='0.0.0.0', port=5000):
    """Check if a port is available for binding."""
    try:
        # Create a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind to the port
        sock.bind((host, port))
        sock.close()
        return True
    except OSError as e:
        if e.errno == 48:  # Address already in use
            return False
        elif e.errno == 13:  # Permission denied
            return False
        else:
            # Other error
            return False
    except Exception:
        return False

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        try:
            # Fallback: get hostname IP
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except:
            return "127.0.0.1"

def test_connection(host='localhost', port=5000, timeout=2):
    """Test if we can connect to a host:port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False