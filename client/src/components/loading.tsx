import React from 'react';
import styled from 'styled-components';
import SyncLoader from 'react-spinners/SyncLoader';

const Container = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: auto;
    margin-bottom: 50px;

    & span {
        padding-top: 20px;
    }
`;

type Props = {
    url: string;
};

const Loading = (props: Props) => {
    const matches = props.url.match(/^https?\:\/\/([^\/?#]+)(?:[\/?#]|$)/i);
    const domain = matches && matches[1]; // domain will be null if no match is found

    return (
        <Container>
            <SyncLoader size={10} />

            <span>Loading privacy policy from {domain}</span>
        </Container>
    );
};

export default Loading;
