# Dataset v3 Lexical Audit

## 1. Vocabulary Overlap
- Unique vocab total: 81
- Ambiguous vocab size: 62
- Not Ambiguous vocab size: 58
- Overlap percentage: 48.15%

## 2. Top TF-IDF Features
### Strongly Associated with Ambiguous
- appropriate number (coef: -2.1275)
- number (coef: -2.1275)
- respond appropriate (coef: -2.1275)
- appropriate (coef: -2.1275)
- utilize 100 (coef: -1.5327)
- utilize 10 (coef: -1.4606)
- reliable (coef: -1.4581)
- users extremely (coef: -1.4581)
- extremely reliable (coef: -1.4581)
- extremely (coef: -1.4581)
- utilize 50 (coef: -1.3942)
- send notification (coef: -1.1438)
- utilize (coef: -1.1264)
- memory (coef: -1.1264)
- shall utilize (coef: -1.1264)
- number tb (coef: -0.8141)
- number ms (coef: -0.7365)
- utilize seconds (coef: -0.7329)
- utilize ms (coef: -0.6999)
- long (coef: -0.6910)

### Strongly Associated with Not Ambiguous
- maximum (coef: 2.9855)
- utilize maximum (coef: 2.9855)
- users 999 (coef: 1.2373)
- uptime (coef: 1.2373)
- 999 (coef: 1.2373)
- 999 uptime (coef: 1.2373)
- shall load (coef: 1.1854)
- respond 100 (coef: 1.0709)
- aes256 (coef: 1.0350)
- respond 10 (coef: 0.9858)
- maximum 100 (coef: 0.9527)
- respond (coef: 0.9433)
- shall respond (coef: 0.9433)
- maximum 10 (coef: 0.9118)
- maximum 50 (coef: 0.8640)
- email (coef: 0.8260)
- email notification (coef: 0.8260)
- send email (coef: 0.8260)
- respond 50 (coef: 0.7792)
- load data (coef: 0.7166)

## 3. Template and Domain Leakage
Domain Distribution by Label:
```
label          Ambiguous  Not Ambiguous
domain                                 
HR              0.426471       0.573529
banking         0.423077       0.576923
cybersecurity   0.443038       0.556962
e-commerce      0.434783       0.565217
education       0.392157       0.607843
finance         0.448276       0.551724
government      0.428571       0.571429
healthcare      0.434783       0.565217
logistics       0.438596       0.561404
manufacturing   0.433333       0.566667
```

## 4. Controlled Pair Contamination
Commonly removed words (Ambiguous cue): {'around': 80, 'number': 65, 'appropriate': 65, 'an': 65, 'be': 56, 'and': 48, 'reliable': 40, 'extremely': 40, '': 16, 'efficient': 16}
Commonly added words (Clear cue): {'a': 96, 'maximum': 80, 'with': 48, 'uptime': 40, '999': 40, '100': 20, '10': 19, 'aes256': 16, 'shall': 16, 'seconds': 13}
