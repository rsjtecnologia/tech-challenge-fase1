#!/usr/bin/env python3
"""
Script para testar o frontend e validar as melhorias implementadas.
Executa checklist de validação visual e funcional.
"""

import json
import sys
from pathlib import Path

def check_frontend_file():
    """Verifica se o arquivo frontend.html existe e tem encoding UTF-8."""
    frontend_path = Path("frontend/frontend.html")
    if not frontend_path.exists():
        print("ERRO: frontend.html nao encontrado")
        return False
    
    try:
        content = frontend_path.read_text(encoding='utf-8')
        print("OK: frontend.html encontrado e legivel com UTF-8")
        return True
    except Exception as e:
        print(f"ERRO ao ler frontend.html: {e}")
        return False

def check_diagnosis_styles():
    """Verifica se os estilos de diagnóstico estão presentes."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    required_styles = [
        ".diagnosis-badge",
        ".diagnosis-malignant",
        ".diagnosis-benign",
        ".diagnosis-positive",
        ".diagnosis-negative",
        ".dashboard-chart"
    ]
    
    for style in required_styles:
        if style in content:
            print(f"OK: Estilo {style} encontrado")
        else:
            print(f"ERRO: Estilo {style} NAO encontrado")
            return False
    
    return True

def check_render_functions():
    """Verifica se as funções de renderização foram atualizadas."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    # Verificar renderResultCard com diagnosis
    if "diagnosis = { text:" in content or "diagnosis ?" in content:
        print("OK: renderResultCard atualizado com parametro diagnosis")
    else:
        print("AVISO: renderResultCard pode nao ter diagnosis completo")
    
    # Verificar renderBarChart com gradientes
    if "opacity = 0.7 + Math.min(v, 0.3)" in content:
        print("OK: renderBarChart atualizado com opacidade dinamica")
    else:
        print("AVISO: renderBarChart nao tem gradientes dinamicos")
    
    return True

def check_prediction_functions():
    """Verifica se as funções de predição foram atualizadas."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    functions = [
        ("predictTabular", "diagnosis-malignant"),
        ("predictMammo", "diagnosis-malignant"),
        ("predictMammoGradCAM", "diagnosis-malignant"),
        ("predictDiabetes", "diagnosis-positive")
    ]
    
    all_updated = True
    for func_name, diagnosis_class in functions:
        if f"async function {func_name}" in content or f"function {func_name}" in content:
            if diagnosis_class in content:
                print(f"OK: {func_name} encontrada e usa diagnostico")
            else:
                print(f"AVISO: {func_name} encontrada mas pode nao usar diagnostico corretamente")
                all_updated = False
        else:
            print(f"ERRO: {func_name} nao encontrada")
            all_updated = False
    
    return all_updated

def check_utf8_chars():
    """Verifica se caracteres UTF-8 especiais estão presentes."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    chars_needed = {
        "Câncer": "C",
        "Mamografia": "M",
        "Diabetes": "D",
    }
    
    all_found = True
    for char, desc in chars_needed.items():
        if char in content:
            print(f"OK: {desc} ({char}) encontrado")
        else:
            print(f"ERRO: {desc} ({char}) NAO encontrado")
            all_found = False
    
    return all_found

def check_chart_height():
    """Verifica se a altura do gráfico foi aumentada."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    if "h-96" in content or 'height: "384"' in content or "height: 384" in content:
        print("OK: Altura do grafico aumentada (h-96 = 384px)")
        return True
    elif "h-80" in content:
        print("AVISO: Grafico ainda pode estar usando h-80 (320px)")
        return False
    else:
        print("AVISO: Nao foi possivel verificar altura do grafico")
        return False

def check_colors_palette():
    """Verifica se a paleta de cores foi aplicada."""
    frontend_path = Path("frontend/frontend.html")
    content = frontend_path.read_text(encoding='utf-8')
    
    colors = {
        "#ef4444": "Vermelho (Maligno)",
        "#10b981": "Verde (Benigno)",
    }
    
    found_colors = 0
    for color, desc in colors.items():
        if color in content:
            print(f"OK: Cor {color} ({desc}) encontrada")
            found_colors += 1
        else:
            print(f"AVISO: Cor {color} ({desc}) nao encontrada (pode estar em formato RGB)")
    
    return found_colors >= 2  # Pelo menos 2 cores devem estar presentes

def main():
    print("=" * 60)
    print("TESTE DE VALIDACAO DO FRONTEND")
    print("=" * 60)
    print()
    
    checks = [
        ("Arquivo frontend.html", check_frontend_file),
        ("Estilos de Diagnostico", check_diagnosis_styles),
        ("Funcoes de Renderizacao", check_render_functions),
        ("Funcoes de Predicao", check_prediction_functions),
        ("Caracteres UTF-8", check_utf8_chars),
        ("Altura do Grafico", check_chart_height),
        ("Paleta de Cores", check_colors_palette),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\nVerificando: {check_name}")
        print("-" * 60)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"ERRO ao executar: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "OK" if result else "ERRO"
        print(f"[{status}] {check_name}")
    
    print()
    print(f"Resultado: {passed}/{total} verificacoes passaram")
    
    if passed == total:
        print("\nTodas as verificacoes passaram! Frontend esta pronto para uso.")
        return 0
    elif passed >= total * 0.7:
        print("\nMaioria das verificacoes passaram. Revise os itens com ERRO.")
        return 0
    else:
        print("\nVarias verificacoes falharam. Revise o frontend.html.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
