# Guía de Deployment en Vercel - Liga MX Predictor

## 📋 Prerrequisitos

1. **Cuenta de Vercel**: Regístrate en [vercel.com](https://vercel.com)
2. **GitHub**: Tu código debe estar en un repositorio de GitHub
3. **Node.js**: Para el frontend (versión 16 o superior)
4. **Python**: Para el backend (versión 3.8 o superior)

## 🚀 Pasos para el Deployment

### 1. Preparar el Repositorio

Asegúrate de que tu proyecto esté en GitHub y que todos los archivos estén committeados:

```bash
git add .
git commit -m "Preparar para deployment en Vercel"
git push origin main
```

### 2. Conectar con Vercel

1. Ve a [vercel.com](https://vercel.com) y haz login
2. Haz clic en "New Project"
3. Conecta tu repositorio de GitHub
4. Selecciona el repositorio `liga_mx_predicto`

### 3. Configuración del Proyecto

Vercel detectará automáticamente la configuración gracias al archivo `vercel.json` que hemos creado:

- **Framework Preset**: Vercel detectará automáticamente React para el frontend
- **Root Directory**: Deja vacío (usará la raíz del proyecto)
- **Build Command**: Se configurará automáticamente
- **Output Directory**: Se configurará automáticamente

### 4. Variables de Entorno (si las necesitas)

Si tu aplicación usa variables de entorno, agrégalas en:
- Vercel Dashboard → Tu Proyecto → Settings → Environment Variables

### 5. Deploy

1. Haz clic en "Deploy"
2. Vercel construirá y desplegará tu aplicación
3. Recibirás una URL como: `https://tu-proyecto.vercel.app`

## 📁 Estructura de Archivos para Vercel

```
liga_mx_predicto/
├── vercel.json              # Configuración de Vercel
├── api/
│   ├── index.py            # Entry point para la API
│   └── requirements.txt    # Dependencias Python
├── frontend/
│   ├── package.json        # Dependencias React
│   └── src/
│       └── App.js          # Frontend actualizado
├── src/
│   └── api/
│       └── main.py         # API FastAPI
└── artifacts/              # Modelos entrenados
```

## 🔧 Configuración Técnica

### Backend (FastAPI)
- **Runtime**: Python 3.9
- **Entry Point**: `api/index.py`
- **Dependencies**: `api/requirements.txt`
- **Max Lambda Size**: 50MB (para los modelos ML)

### Frontend (React)
- **Framework**: Create React App
- **Build Command**: `npm run build`
- **Output Directory**: `frontend/build`

### Rutas
- `/api/*` → Backend FastAPI
- `/*` → Frontend React

## 🐛 Solución de Problemas

### Error: "Module not found"
- Verifica que todas las dependencias estén en `api/requirements.txt`
- Asegúrate de que los imports en `api/index.py` sean correctos

### Error: "CORS"
- El frontend ya está configurado para usar rutas relativas en producción
- Las URLs de CORS en `main.py` incluyen dominios de Vercel

### Error: "Model files not found"
- Verifica que los archivos en `artifacts/` estén committeados
- Los modelos deben estar en el repositorio para ser accesibles

### Error: "Lambda size exceeded"
- Los modelos ML pueden ser grandes
- Considera usar modelos más pequeños o optimizados
- El límite actual es 50MB

## 📊 Monitoreo

Una vez desplegado, puedes:
- Ver logs en tiempo real en Vercel Dashboard
- Monitorear el rendimiento
- Configurar alertas
- Ver analytics de uso

## 🔄 Actualizaciones

Para actualizar tu aplicación:
1. Haz cambios en tu código local
2. Commit y push a GitHub
3. Vercel desplegará automáticamente la nueva versión

## 💡 Tips Adicionales

1. **Dominio Personalizado**: Puedes conectar tu propio dominio en Vercel
2. **Preview Deployments**: Cada PR crea un deployment de preview
3. **Environment Variables**: Usa variables de entorno para configuraciones sensibles
4. **Analytics**: Habilita Vercel Analytics para métricas de rendimiento

## 🆘 Soporte

Si tienes problemas:
1. Revisa los logs en Vercel Dashboard
2. Verifica la configuración en `vercel.json`
3. Consulta la [documentación de Vercel](https://vercel.com/docs)
4. Revisa los issues en el repositorio

¡Tu aplicación de predicciones de Liga MX estará lista para el mundo! ⚽
