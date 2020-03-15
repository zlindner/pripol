import React from 'react';
import styled from 'styled-components';
import { ReactComponent as Analyze } from '../assets/analyze.svg';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
`;

const Search = styled.div`
    width: 800px;
    display: flex;
    align-items: center;
    padding: 10px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);

    & input {
        width: calc(100% - 40px);
        height: 50px;
        margin-right: 20px;
        font-size: 24px;
        border: none;
    }

    & svg {
        width: 40px;
        height: 40px;
        cursor: pointer;
    }
`;

const Landing = () => {
    return (
        <Container>
            <Search>
                <input type='text' placeholder="Analyze a website's privacy policy" autoFocus={true} />

                <Analyze />
            </Search>
        </Container>
    );
};

export default Landing;
