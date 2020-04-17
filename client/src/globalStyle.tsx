import { createGlobalStyle } from 'styled-components';

export default createGlobalStyle`
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }

    a {
        color: #000000;
        text-decoration: none;
    }

    a:active {
        color: #000000;
    }

    a:visited {
        color: #000000;
    }
`;
