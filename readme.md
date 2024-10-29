# Descargar y ejecutar un repositorio desde GitHub

## Paso 1: Clonar el repositorio desde GitHub

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

## Paso 2: Configurar el entorno

### Instalar Python 3.11

Asegúrate de tener Python 3.11 instalado. Puedes verificar tu versión de Python con:

```bash
python --version
```

### Crear un entorno virtual con los requerimientos

```bash
python -m venv env
source env/bin/activate # En Windows usa `env\Scripts\activate`
pip install -r requirements.txt
```

### Ejectutar la aplicación de FastAPI

Usa este comando para ejectutar la aplicación una vez que tengas los requerimientos instalados.

```bash
uvicorn main:app --reload
```

## Opcional: Ejectutar el proyecto usando Docker

### Contruir la imagen de docker

Asegúrate de tener Docker instalado y corriendo. Luego, desde la raíz del repositorio, construye la imagen:

```bash
docker build -t nombre_que_asignes .
```

### Ejecuta el contenedor

```bash
docker run -p 8000:8000 nombre_de_tu_imagen
```
